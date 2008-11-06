#!/bin/sh
#$Id: t0inject.sh,v 1.15 2008/11/06 00:58:48 loizides Exp $

. /etc/init.d/functions

# base dir for code
export SMT0_BASE_DIR=/nfshome0/smpro/sm_scripts_cvs/inject

# db config file
export SMT0_CONFIG=/nfshome0/smpro/configuration/db.conf

#local run dir
export SMT0_LOCAL_RUN_DIR=/nfshome0/smpro/t0inject

if test -d "/store/injectworker"; then
    export SMT0_BASE_DIR=/opt/injectworker/inject
    export SMT0_CONFIG=/opt/injectworker/.db.conf
    export SMT0_LOCAL_RUN_DIR=/store/injectworker
    if test -e "/opt/copyworker/t0_control.sh"; then
        source /opt/copyworker/t0_control.sh status > /dev/null
    fi
fi

if [ ! -d $SMT0_BASE_DIR ]; then
    echo "SMT0_BASE_DIR ($SMT0_BASE_DIR) does not exist or is no directory"
    exit
fi

# main perl script
SMT0_IW=$SMT0_BASE_DIR/InjectWorker.pl

if [ ! -x $SMT0_IW ]; then
    echo "SMT0_IW ($SMT0_IW)does not exist or is not executable"
    exit
fi

# directory to monitor
SMT0_MONDIR=/store/global/log

if test -n "$SM_STORE"; then
    SMT0_MONDIR=$SM_STORE/global/log
fi

if [ ! -d $SMT0_MONDIR ]; then
    echo "SMT0_MONDIR ($SMT0_MONDIR) does not exist or is no directory"
    exit
fi

if [ ! -r $SMT0_CONFIG ]; then
    echo "SMT0_CONFIG ($SMT0_CONFIG) can not be read"
    exit
fi

#exported variables
export SM_NOTIFYSCRIPT=$SMT0_BASE_DIR/sendNotification.sh
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
            ${SMT0_LOCAL_RUN_DIR}/logs ${SMT0_CONFIG} $inst > `hostname`.$$ 2>&1 &
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
    ls /tmp/.2*-`hostname`-*.log.lock 2>/dev/null
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
        status
        ;;
    stop)
        stop
        status
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
