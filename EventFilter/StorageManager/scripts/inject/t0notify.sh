#!/bin/sh
#$Id: t0notify.sh,v 1.1 2010/06/01 14:33:59 babar Exp $

. /etc/init.d/functions

# base dir for code
export SMT0_BASE_DIR=/nfshome0/smpro/sm_scripts_cvs/inject

# db config file
export SMT0_CONFIG=/nfshome0/smpro/configuration/db.conf

#local run dir
export SMT0_LOCAL_RUN_DIR=/nfshome0/smpro/t0inject

if test -d "/opt/injectworker"; then
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
SMT0_NW=$SMT0_BASE_DIR/NotifyWorker.pl

if [ ! -x $SMT0_NW ]; then
    echo "SMT0_NW ($SMT0_NW)does not exist or is not executable"
    exit
fi

# directory to monitor
SMT0_MONDIR=$SMT0_LOCAL_RUN_DIR/logs

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

    echo "Starting $SMT0_NW"
    nohup ${SMT0_NW} ${SMT0_MONDIR} ${SMT0_LOCAL_RUN_DIR}/logs \
        ${SMT0_CONFIG} > `hostname`.$$ 2>&1 &
    sleep 1
}

stop(){
    for pid in `ps ax | grep ${SMT0_NW} | grep -v grep | cut -b1-6 | tr -d " "`; do
	echo "Attempting to stop worker with pid $pid"
	kill -s 15 $pid
    done
    rm -f ${SMT0_LOCAL_RUN_DIR}/workdir/`hostname`.*
    ls /tmp/.2*-`hostname`-*.log.lock 2>/dev/null
}

status(){
    for pid in `ps ax | grep ${SMT0_NW} | grep -v grep | cut -b1-6 | tr -d " "`; do
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
