#!/bin/sh
# $Id: InjectWorker.sh,v 1.4 2008/04/28 11:11:35 loizides Exp $
#
#  ./InjectWorker.sh directory/file scripts [logdir errordir]
#
# or parse environment
#
#  $SM_IWINPUT for directory/file
#  $SW_IWSCRIPT for script
#  $SM_IWLOGDIR for logdir
#  $SM_IWERRORDIR for errordir
#

function exit_on_trap()
{
    echo -n "Stopping $0 with PID $$ at " >> $IW_LOGFILE
    echo `date` >> $IW_LOGFILE
    exit 0;
}

function getLogFileName()
{
    local hname=`hostname | cut -d. -f1`
    local dname=`date "+%Y%m%d"`
    if test -n "$SMIW_RUNNUM"; then
	echo "$dname-$hname-iw-${SMIW_RUNNUM}.log"
    else
	echo "$dname-$hname-iw-$$.log"
    fi
}

# test for IW_LOGDIR
if test -n "$3"; then
    IW_LOGDIR=$3
else
    IW_LOGDIR=$SM_IWLOGDIR
fi
if test -z "$IW_LOGDIR"; then
    IW_LOGDIR=/tmp
fi
IW_LOGFILE=$IW_LOGDIR/`getLogFileName`
echo -n "Starting $0 with PID $$ at " >> $IW_LOGFILE
echo `date` >> $IW_LOGFILE

# set craceful exit
trap exit_on_trap 1 2 9 15

# test for IW_INPUT
if test -n "$1"; then
    IW_INPUT=$1
else
    IW_INPUT=$SM_IWINPUT
fi
if test -z "$IW_INPUT"; then
    echo "ERROR $0: no input given." >> $IW_LOGFILE
    exit 123;
fi
if test -d "$IW_INPUT"; then
    IW_DIRMODE=1;
elif test -e "$IW_INPUT"; then
    IW_DIRMODE=0;
else
    echo "ERROR $0: $IW_INPUT is neither directory nor file." >> $IW_LOGFILE
    exit 124;
fi

# test for IW_SCRIPT
if test -n "$2"; then
    IW_SCRIPT=$2
else
    IW_SCRIPT=$SM_IWSCRIPT;
fi
if test -z "$IW_SCRIPT"; then
    echo "ERROR $0: no worker script given." >> $IW_LOGFILE
    exit 125;
fi
if ! test -x "$IW_SCRIPT"; then
    echo "ERROR $0: Script $IW_SCRIPT is not found." >> $IW_LOGFILE
    exit 125;
fi

# test for IW_ERRORDIR
if test -n "$4"; then
    IW_ERRORDIR=$4
else
    IW_ERRORDIR=$SM_IWERRORDIR
fi
if test -n "$IW_ERRORDIR"; then
    if ! test -d "$IW_ERRORDIR"; then
        mkdir -p $IW_ERRORDIR
        if ! test -d "$IW_ERRORDIR"; then
            echo "ERROR $0: Error dir $IW_ERRORDIR can not be created." >> $IW_LOGFILE
            exit 125;
           
        fi
    fi
fi

IW_SLEEPTIME=15
#IW_PROCESSFILE=true

COUNTER=0
# start main loop
while test 1 -gt 0; do
    IFILE=""
    COUNTER=`expr $COUNTER + 1`
    if test "$IW_DIRMODE" = "1"; then
        for i in `ls -f $IW_INPUT/*.notify 2>/dev/null`; do
            IFILE=$i
            break;
        done

        if test -z "$IFILE"; then
            sleep $IW_SLEEPTIME
            if test $COUNTER -gt 1000; then
                echo "Info $0: Still alive at `date`" >> $IW_LOGFILE
                COUNTER=0
            fi
            continue;
        fi
        COUNTER=0
    else
        IFILE=$IW_INPUT;
    fi

    #ignore common signals
    trap '' 1 2 9 15

    #set log file in case date changed 
    IW_LOGFILE=$IW_LOGDIR/`getLogFileName`

    echo -n "Info $0: Working on file $IFILE at " >> $IW_LOGFILE
    echo `date` >> $IW_LOGFILE

    IFILEDIR=`dirname  $IFILE`
    IFILEBAS=`basename $IFILE .notify`

    TFILE=${IFILEDIR}/${IFILEBAS}.notifying
    WFILE=${IFILEDIR}/${IFILEBAS}.notified

    #rename to notifying
    mv $IFILE $TFILE
    touch $WFILE

    #process file
    if test -n "$IW_PROCESSFILE"; then
        cat $TFILE | while read line; do 
            eval "$line"
            $IW_SCRIPT >> $IW_LOGFILE 2>&1
            if test "$?" -eq 0; then
                echo $line > $WFILE
            fi
        done
    else
        $IW_SCRIPT $TFILE $WFILE  >> $IW_LOGFILE 2>&1
    fi        

    #test if WFILE has equal content
    test=`diff -q $WFILE $TFILE 1>/dev/null`;
    if test "$?" -eq 0; then
        rm -f $WFILE $TFILE
    else
        echo "ERROR $0: Difference in files $WFILE $TFILE, not removed:"  >> $IW_LOGFILE
        echo "---> `date`"  >> $IW_LOGFILE
        diff $WFILE $TFILE >> $IW_LOGFILE 2>&1
        if test -n "$IW_ERRORDIR"; then
            echo " $WFILE $TFILE, moved to $IW_LOGDIR."  >> $IW_LOGFILE
            mv $WFILE $TFILE $IW_ERRORDIR
        fi
    fi

    echo -n "Info $0: Finished file $IFILE at " >> $IW_LOGFILE
    echo `date` >> $IW_LOGFILE

    #reset signals
    trap exit_on_trap 1 2 9 15

    if test "$IW_DIRMODE" = "0"; then
        break;
    fi
done
