#!/bin/sh
#$Id: t0inject.sh,v 1.8 2008/05/19 19:41:37 loizides Exp $

. /etc/init.d/functions

export SMT0_BASE_DIR=/nfshome0/smpro/sm_scripts_cvs/inject
if [ ! -d $SMT0_BASE_DIR ]; then
    echo "SMT0_BASE_DIR does not exist or is no directory"
    exit
fi

SMT0_IW=$SMT0_BASE_DIR/InjectWorker.pl
if [ ! -x $SMT0_IW ]; then
    echo "SMT0_IW does not exist or is not executable"
    exit
fi

SMT0_MONDIR=/store/global/log
if test -n "$SM_STORE"; then
    SMT0_MONDIR=$SM_STORE/global/log
fi
if [ ! -d $SMT0_MONDIR ]; then
    echo "SMT0_MONDIR does not exist or is no directory"
    exit
fi

#local run dir
export SMT0_LOCAL_RUN_DIR=/nfshome0/smpro/t0inject

#exported variables
export SM_NOTIFYSCRIPT=$SMT0_BASE_DIR/sendNotification.sh
#export SM_NOTIFYSCRIPT=/nfshome0/cmsprod/TransferTest/injection/sendNotification.sh
export SM_HOOKSCRIPT=$SMT0_BASE_DIR/sm_hookscript.pl

#
# Define rules to start and stop daemons
#
start(){
    #
    # Setting up environment
    #
    mkdir -p ${SMT0_LOCAL_RUN_DIR}/logs
    mkdir -p ${SMT0_LOCAL_RUN_DIR}/done
    mkdir -p ${SMT0_LOCAL_RUN_DIR}/workdir

    cd ${SMT0_LOCAL_RUN_DIR}/workdir

    #running with four instances should be enough
    for i in `seq 1 4`; do
        inst=`expr $i - 1`
        export SMIW_RUNNUM=$inst
        echo "Starting $SMT0_IW instance $inst"
        nohup ${SMT0_IW} ${SMT0_MONDIR} ${SMT0_LOCAL_RUN_DIR}/done \
            ${SMT0_LOCAL_RUN_DIR}/logs $inst > `hostname`.$$ 2>&1 &
        sleep 1
    done
    echo
}

stop(){
    for pid in `ps ax | grep ${SMT0_IW} | grep -v grep | cut -b1-6 | tr -d " "`; do
	echo "Attempting to stop worker with pid $pid"
	kill -s 15 $pid
    done
    rm -f ${SMT0_LOCAL_RUN_DIR}/workdir/`hostname`.*
}

status(){
    for pid in `ps ax | grep ${SMT0_IW} | grep -v grep | cut -b1-6 | tr -d " "`; do
	echo `/bin/ps $pid | grep $pid`
    done
}

cleanup(){
    find ${SMT0_LOCAL_RUN_DIR}/logs -type f -name "*.log*" -exec rm -f {} \;
    find ${SMT0_LOCAL_RUN_DIR}/done -type f -name "*.log*" -exec rm -f {} \;
    find ${SMT0_LOCAL_RUN_DIR}/workdir -type f -name "*.*" -exec rm -f {} \;
}


# See how we were called.
case "$1" in
    start)
	start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    cleanup)
        cleanup
        ;;
    *)
        echo $"Usage: $0 {start|stop|status|cleanup}"
        RETVAL=1
esac
