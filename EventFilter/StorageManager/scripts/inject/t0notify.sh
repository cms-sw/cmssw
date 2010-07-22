#!/bin/sh
#$Id: t0notify.sh,v 1.4 2010/07/21 09:58:29 babar Exp $

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

    ( # Double-fork
        cd ${SMT0_LOCAL_RUN_DIR}/workdir
        exec > `hostname`.$$
        exec 2>&1
        exec </dev/null
        ${SMT0_NW} ${SMT0_MONDIR} ${SMT0_LOCAL_RUN_DIR}/logs ${SMT0_CONFIG} &
    )
}

stop(){
    pkill -15 -f -P 1 -u smpro $SMT0_NW
    rm -f ${SMT0_LOCAL_RUN_DIR}/workdir/`hostname`.*
    if [ -f /tmp/.NotifyWorker.lock ]; then
        echo 'Lockfile is still there!'
        exit 1
    fi
}

status(){
    pgrep -l -f -P 1 -u smpro $SMT0_NW
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
        test ! status
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
    ;;
esac
