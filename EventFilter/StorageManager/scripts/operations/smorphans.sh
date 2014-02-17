#!/bin/bash
# $Id: smorphans.sh,v 1.5 2009/04/29 15:22:24 jserrano Exp $

# This script iterates over all files in the list of hosts provided and calls
# injection script with --check option to show a status report.
#
# Example to automate cleanup on each SM node for closed files older than 30 days:
# ./smorphans.sh --clean --closedfiles --age=30 --host=`hostname` --FILES_ALL
#
# TODO: Embed DB query code to avoid calling external script for every file,
# will improve performance dramatically: group all queries in a large one.
#
# Have a look at the configuration file
CFG_FILE=/opt/smops/smorphans.cfg 
source $CFG_FILE
if [ ! -r $HOSTLIST ]
then
    echo "ERROR: File $HOSTLIST missing!"
    exit 1
fi

DO=""
DO_REPORT=""
DO_FULL_REPORT=""
DO_CLEAN=""
DO_OPEN=""
DO_CLOSED=""
DO_FILES=""
DO_FILES_EMU=""
DO_FILE_AGE=""
DO_FILES_DELETED=""
DO_FILES_CREATED=""
DO_FILES_INJECTED=""
DO_FILES_INJECTED=""
DO_FILES_TRANS_COPIED=""
DO_FILES_TRANS_CHECKED=""
DO_FILES_TRANS_INSERTED=""
DO_FILES_DELETED=""

report() {
    FILES_CREATED=0
    FILES_INJECTED=0
    FILES_TRANS_NEW=0
    FILES_TRANS_COPIED=0
    FILES_TRANS_CHECKED=0
    FILES_TRANS_INSERTED=0
    FILES_DELETED=0
    if [[ "$DO_FILE_AGE" -gt 0 ]]; then AGE_QUERY="-mtime +$DO_FILE_AGE"; fi
#    for host in $( cat $HOSTLIST ); do
    for host in $( if [[ $DO_HOST ]]; then echo $DO_HOST; else cat $HOSTLIST ; fi ); do
        echo ----- $host -----
#        for file in $( ssh $host "find $1 $AGE_QUERY -name '*.dat' 2> /dev/null" ); do
        for file in $( ssh $host "find $1 $AGE_QUERY -name '*.dat' " ); do
            basename=`basename $file`
            status=`$CHECK_INJECTION_CMD$basename | cut -d: -f1`
            if [ $DO_FULL_REPORT ]; then
#                filedata=`ssh $host "find $file -printf '%k, %TD' 2> /dev/null"`
                filedata=`ssh $host "find $file -printf '%TD, %kK' "`
                echo "$host, $filedata, $status, $file"
            fi
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
        echo Directory size
#        ssh $host "du -h $1 2> /dev/null"
        ssh $host "du -h $1 "
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
    if [[ "$DO_FILE_AGE" -gt 0 ]]; then AGE_QUERY="-mtime +$DO_FILE_AGE"; fi
    for host in $( cat $HOSTLIST ); do
#        for file in $( ssh $host "find $1 $AGE_QUERY -name '*.dat' 2> /dev/null" ); do
        for file in $( ssh $host "find $1 $AGE_QUERY -name '*.dat'" ); do
            basename=`basename $file`
            status=`$CHECK_INJECTION_CMD$basename`
            case $status in
                FILES_CREATED*)
                    if [ $DO_FILES_CREATED ]; then
                        delete_file $host $file
                    fi
                    ;;
                FILES_INJECTED*) 
                    if [ $DO_FILES_INJECTED ]; then
                        delete_file $host $file
                    fi
                    ;;
                FILES_TRANS_NEW*)
                    if [ $DO_FILES_TRANS_NEW ]; then
                        delete_file $host $file
                    fi
                    ;;

                FILES_TRANS_COPIED*)
                    if [ $DO_FILES_TRANS_COPIED ]; then
                        delete_file $host $file
                    fi
                    ;;

                FILES_TRANS_CHECKED*)
                    if [ $DO_FILES_TRANS_CHECKED ]; then
                        delete_file $host $file
                    fi
                    ;;

                FILES_TRANS_INSERTED*)
                    if [ $DO_FILES_TRANS_INSERTED ]; then
                        delete_file $host $file
                    fi
                    ;;

                FILES_DELETED*)
                    if [ $DO_FILES_DELETED ]; then
                        delete_file $host $file
                    fi
                    ;;
                *)
                    echo Internal error: unkown inject status;;
            esac
        done
    done
}

delete_file() {
# TODO: record some flag in DB before delete.
    if [ $DO_FILES_EMU ]; then
       echo ssh $1 "sudo chmod 666 $2; sudo rm -f $2"
       echo
    else
       `ssh $1 "sudo chmod 666 $2; sudo rm -f $2"`
    fi
}

show_usage() {
    cat << EOF

$0 usage:

   -h                 print this help
   --help             print this help

Actions:
   --report           print report
   --fullreport       print report with all file names
   --clean            clean files

Options:
   --age=X            act only in files older than X days
   --emulate          emulate, do not remove any files
   --host=hostname    provide hostname to run

File types:
Actions only applied to selected file type. You must select one file type at least.

   --closedfiles      process closed files
   --openfiles        process open files

File status:
Only selected file status will be cleaned. You must select one file status at least.

    --FILES_CREATED
    --FILES_INJECTED
    --FILES_TRANS_NEW
    --FILES_TRANS_COPIED
    --FILES_TRANS_CHECKED
    --FILES_TRANS_INSERTED
    --FILES_DELETED
    --FILES_ALL


Configure directories, list of PCs, and external calls in $CFG_FILE

EOF
    exit -1
}

# Entry point
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
                --fullreport)
                    DO=1
                    DO_REPORT=1
                    DO_FULL_REPORT=1
                    shift ;;
                --clean)
                    DO=1
                    DO_CLEAN=1
                    shift ;;
                --emulate)
                    DO_FILES_EMU=1
                    shift ;;
                --openfiles)
                    DO_OPEN=1
                    shift ;;
                --closedfiles)
                    DO_CLOSED=1
                    shift ;;
                --age=*)
                    DO_FILE_AGE=$(( `echo $ARG | cut -d= -f2` + 0 ))
                    shift;;
                --host=*)
                    DO_HOST=`echo $ARG | cut -d= -f2`
                    shift;;
                --FILES_CREATED)
                    DO_FILES=1
                    DO_FILES_CREATED=1
                    shift ;;
                --FILES_INJECTED)
                    DO_FILES=1
                    DO_FILES_INJECTED=1
                    shift ;;
                --FILES_TRANS_NEW)
                    DO_FILES=1
                    DO_FILES_INJECTED=1
                    shift ;;
                --FILES_TRANS_COPIED)
                    DO_FILES=1
                    DO_FILES_TRANS_COPIED=1
                    shift ;;
                --FILES_TRANS_CHECKED)
                    DO_FILES=1
                    DO_FILES_TRANS_CHECKED=1
                    shift ;;
                --FILES_TRANS_INSERTED)
                    DO_FILES=1
                    DO_FILES_TRANS_INSERTED=1
                    shift ;;
                --FILES_DELETED)
                    DO_FILES=1
                    DO_FILES_DELETED=1
                    shift ;;
                --FILES_ALL)
                    DO_FILES=1
                    DO_FILES_DELETED=1
                    DO_FILES_CREATED=1
                    DO_FILES_INJECTED=1
                    DO_FILES_INJECTED=1
                    DO_FILES_TRANS_COPIED=1
                    DO_FILES_TRANS_CHECKED=1
                    DO_FILES_TRANS_INSERTED=1
                    DO_FILES_DELETED=1
                    shift ;;

                *)
                    echo Syntax error: unkown parameter $ARG
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
    echo "Missing parameter: you must select 1 type of files at least (open/closed)."
    show_usage
fi

if [ $DO_REPORT ]; then
    if [ $DO_OPEN ]; then report "$CHECK_PATH_OPEN"; fi
    if [ $DO_CLOSED ]; then report "$CHECK_PATH_CLOSED"; fi
fi

if [ $DO_CLEAN ] ; then
    if [ $DO_FILES ]; then
        if [ $DO_OPEN ]; then clean "$CHECK_PATH_OPEN"; fi
        if [ $DO_CLOSED ]; then clean "$CHECK_PATH_CLOSED"; fi
    else
        echo "Missing parameter: You must select 1 file status at least."
        show_usage
    fi
fi

exit
}
