#!/bin/bash
# $Id: check_left_files.sh,v 1.3 2008/07/03 10:46:58 loizides Exp $

# This script iterates over all files in the list of hosts provided and calls
# injection script with --check option to show a status report.

# Have a look at the configuration file
CFG_FILE=smorphans.cfg 
source $CFG_FILE
if [ ! -r $HOSTLIST ]
then
    echo "ERROR: File $HOSTLIST missing!"
    exit 1
fi


report() {
echo report started
    FILES_CREATED=0
    FILES_INJECTED=0
    FILES_TRANS_NEW=0
    FILES_TRANS_COPIED=0
    FILES_TRANS_CHECKED=0
    FILES_TRANS_INSERTED=0
    FILES_DELETED=0

    for host in $( cat $HOSTLIST ); do
        echo ----- $host -----
        echo Directory size
        ssh $host "du -h $1"
        for file in $( ssh $host "find $1 -name '*.dat'" ); do
            #echo $file;
            basename=`basename $file`
            status=`$CHECK_INJECTION_CMD$basename`
            #echo $status
            case $status in
                FILES_CREATED*)        FILES_CREATED=`expr $FILES_CREATED + 1`;;
                FILES_INJECTED*)       FILES_INJECTED=`expr $FILES_INJECTED + 1`;;
                FILES_TRANS_NEW*)      FILES_TRANS_NEW=`expr $FILES_TRANS_NEW + 1`;;
                FILES_TRANS_COPIED*)   FILES_TRANS_COPIED=`expr $FILES_TRANS_COPIED + 1`;;
                FILES_TRANS_CHECKED*)  FILES_TRANS_CHECKED=`expr $FILES_TRANS_CHECKED + 1`;;
                FILES_TRANS_INSERTED*) FILES_TRANS_INSERTED=`expr $FILES_TRANS_INSERTED + 1`;;
                FILES_DELETED*)        FILES_DELETED=`expr $FILES_DELETED + 1`;;
                *)                     echo Internal error: unkown inject status;;
            esac
        done
	echo File categories
        echo FILES_CREATED $FILES_CREATED
        echo FILES_INJECTED $FILES_INJECTED
        echo FILES_TRANS_NEW $FILES_TRANS_NEW
        echo FILES_TRANS_COPIED $FILES_TRANS_COPIED
        echo FILES_TRANS_CHECKED $FILES_TRANS_CHECKED
        echo FILES_TRANS_INSERTED $FILES_TRANS_INSERTED
        echo FILES_DELETED $FILES_DELETED
        echo
    done
}

clean() {
    cat << EOF
I am supposed to clean in $1
Some specific action has to be defined for each category:

                FILES_CREATED
                FILES_INJECTED
                FILES_TRANS_NEW
                FILES_TRANS_COPIED
                FILES_TRANS_CHECKED
                FILES_TRANS_INSERTED
                FILES_DELETED

EOF


    for host in $( cat $HOSTLIST ); do
        for file in $( ssh $host "find $1 -name '*.dat'" ); do
            basename=`basename $file`
            status=`$CHECK_INJECTION_CMD$basename`
            # TODO: fill actions for each category bellow
            # $file contains full path name
            # $host contains host name
            # one can easily call ssh $host "sudo chmod 666 $file; sudo rm -f $file"
            case $status in
                FILES_CREATED*)        ;;
                FILES_INJECTED*)       ;;
                FILES_TRANS_NEW*)      ;;
                FILES_TRANS_COPIED*)   ;;
                FILES_TRANS_CHECKED*)  ;;
                FILES_TRANS_INSERTED*) ;;
                FILES_DELETED*)        ;;
                *)                     echo Internal error: unkown inject status;;
            esac
        done
    done
}

show_usage() {
    cat << EOF

$0 usage:
   -h                 print this help
   --help             print this help
   --report           print a report
   --clean            clean files
   --closedfiles      process closed files
   --openfiles        process open files

You must select 1 file type and 1 action at least
Configure directories, list of PCs, and external calls in $CFG_FILE

EOF
    exit -1
}

# Entry point

if [[ $# -lt 1 ]] ; then
 echo "Syntax error: missing parameter"
 echo
 show_usage
fi

for ARG in "$@"; do
    case $ARG in
        -*) true ;
            case $ARG in
                -h)
                    show_usage;;
                --help)
                    show_usage;;
                --report)
                    DO=1
                    DO_REPORT=1
                    shift ;;
                --clean)
                    DO=1
                    DO_CLEAN=1
                    shift ;;
                --openfiles)
                    DO_OPEN=1
                    shift ;;
                --closedfiles)
                    DO_CLOSED=1
                    shift ;;
                *)
                    echo Syntax error: unkown parameter
                    show_usage;;
            esac
        ;;
    esac
done

if [ ! $DO ]; then
    echo "Syntax error: you must select 1 action at least"
    show_usage
fi
if [ ! $DO_OPEN ] && [ ! $DO_CLOSED ]; then
    echo "Syntax error: you must select 1 type of files at least (open/closed)"
    show_usage
fi

if [ $DO_REPORT ]; then
    if [ $DO_OPEN ]; then report "$CHECK_PATH_OPEN"; fi
    if [ $DO_CLOSED ]; then report "$CHECK_PATH_CLOSED"; fi
fi

if [ $DO_CLEAN ] ; then
    if [ $DO_OPEN ]; then clean "$CHECK_PATH_OPEN"; fi
    if [ $DO_CLOSED ]; then clean "$CHECK_PATH_CLOSED"; fi
fi

exit
}
