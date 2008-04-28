#!/bin/sh
#$Id:$

. /etc/init.d/functions

SMT0_BASE_DIR=/nfshome0/smpro/sm_scripts_cvs/inject
if [ ! -d $SMT0_BASE_DIR ]; then
    echo "SMT0_BASE_DIR does not exist or is no directory"
    exit
fi

SMT0_IW=$SMT0_BASE_DIR/InjectWorker.sh
if [ ! -x $SMT0_IW ]; then
    echo "SMT0_IW does not exist or is not executable"
    exit
fi

SMT0_IWS=$SMT0_BASE_DIR/injectIntoDB.pl
if [ ! -x $SMT0_IWS ]; then
    echo "SMT0_IWS does not exist or is not executable"
    exit
fi

SMT0_LOCAL_RUN_DIR=/nfshome0/smpro/t0inject
SMT0_MONDIR=/store/global/mbox
if test -n "$SM_STORE"; then
    SMT0_MONDIR=$SM_STORE/global/mbox
fi
if [ ! -d $SMT0_MONDIR ]; then
    echo "SMT0_MONDIR does not exist or is no directory"
    exit
fi

#
# Define rules to start and stop daemons
#
start(){
    #
    # Setting up environment
    #
    mkdir -p ${SMT0_LOCAL_RUN_DIR}/logs
    mkdir -p ${SMT0_LOCAL_RUN_DIR}/problem
    mkdir -p ${SMT0_LOCAL_RUN_DIR}/workdir

    cd ${SMT0_LOCAL_RUN_DIR}/workdir

    #running with one instance should be enough
    export SMIW_RUNNUM=1
    echo -n $"Starting $SMT0_IW"
    nohup ${SMT0_IW} ${SMT0_MONDIR} ${SMT0_IWS} ${SMT0_LOCAL_RUN_DIR}/logs ${SMT0_LOCAL_RUN_DIR}/problem > `hostname`.$$ 2>&1 &
    sleep 3
    echo
}

stop(){
    for pid in `/bin/ps ax | grep ${SMT0_IW} | grep -v grep | cut -d' ' -f 1`; do
	kill $pid
    done
    rm -f ${SMT0_LOCAL_RUN_DIR}/workdir/`hostname`.*
}

status(){
    for pid in `/bin/ps ax | grep ${SMT0_IW} | grep -v grep | cut -d' ' -f 1`; do
	echo `/bin/ps $pid | grep $pid`
    done
}

cleanup(){
    find ${SMT0_LOCAL_RUN_DIR}/logs -type f -name "*.log*" -exec rm -f {} \;
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
