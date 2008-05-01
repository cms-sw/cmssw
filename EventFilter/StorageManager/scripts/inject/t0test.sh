#!/bin/sh
#$Id: t0test.sh,v 1.1 2008/04/29 21:46:55 loizides Exp $

. /etc/init.d/functions

SMT0_BASE_DIR=$CMSSW_BASE/src/EventFilter/StorageManager/scripts/inject
if [ ! -d $SMT0_BASE_DIR ]; then
    SMT0_BASE_DIR=$CMSSW_RELEASE_BASE/src/EventFilter/StorageManager/scripts/inject
    if [ ! -d $SMT0_BASE_DIR ]; then
        echo "SMT0_BASE_DIR does not exist or is no directory"
        exit
    fi
fi

SMT0_IW=$SMT0_BASE_DIR/InjectWorker.sh
if [ ! -x $SMT0_IW ]; then
    echo "SMT0_IW does not exist or is not executable"
    exit
fi

SMT0_IWS=$SMT0_BASE_DIR/injectDummy.pl
if [ ! -x $SMT0_IWS ]; then
    echo "SMT0_IWS does not exist or is not executable"
    exit
fi

SMT0_MONDIR=/tmp/$USER/sm/mbox

SMT0_LOCAL_RUN_DIR=/tmp/t0inject
mkdir -p $SMT0_LOCAL_RUN_DIR

#
# Define rules to start and stop daemons
#
start(){
    #
    # Setting up environment
    #
    mkdir -p ${SMT0_LOCAL_RUN_DIR}/logs
    mkdir -p ${SMT0_LOCAL_RUN_DIR}/error
    mkdir -p ${SMT0_LOCAL_RUN_DIR}/keep
    mkdir -p ${SMT0_LOCAL_RUN_DIR}/workdir

    cd ${SMT0_LOCAL_RUN_DIR}/workdir

    #running with one instance should be enough
    export SMIW_RUNNUM=1

    echo -n $"Starting $SMT0_IW"
    nohup ${SMT0_IW} ${SMT0_MONDIR} ${SMT0_IWS} ${SMT0_LOCAL_RUN_DIR}/logs \
          ${SMT0_LOCAL_RUN_DIR}/error ${SMT0_LOCAL_RUN_DIR}/keep > `hostname`.$$ 2>&1 &
    sleep 3
    echo
}

stop(){
    for pid in `ps ax | grep ${SMT0_IW} | grep -v grep | cut -b1-6 | tr -d " "`; do
	echo `/bin/ps $pid | grep $pid`
	kill -9 $pid
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
