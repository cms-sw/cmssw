#!/bin/sh
# $Id: setup_sm.sh,v 1.39 2009/07/20 19:31:07 loizides Exp $

if test -e "/etc/profile.d/sm_env.sh"; then 
    source /etc/profile.d/sm_env.sh;
fi

#
# variables
#

store=/store
if test -n "$SM_STORE"; then
    store=$SM_STORE
fi
lookarea=$store/lookarea
if test -n "$SM_LOOKAREA"; then
    lookarea=$SM_LOOKAREA
fi

cmhost=srv-C2C07-20

hname=`hostname | cut -d. -f1`;
nname="node"`echo $hname | cut -d- -f3` 
case $hname in
    cmsdisk0)
        nname="nottobeused"
        ;;
    srv-S2C17-01)
        nname=node_cms-tier0-stage
        ;;
    srv-C2D05-02)
        nname=node_cmsdisk1
        ;;
    *)
        ;;
esac

t0control="~cmsprod/$nname/t0_control.sh";
if test -e "/opt/copyworker/t0_control.sh"; then
    t0control="/opt/copyworker/t0_control.sh"
fi

t0cmcontrol="~cmsprod/TransferTest/old_t0_transferstatusworker.sh_old";
if test -e "/opt/copymanager/t0_control.sh"; then
    t0cmcontrol="/opt/copymanager/t0_control.sh"
fi

t0inject="~smpro/scripts/t0inject.sh";
if test -e "/opt/injectworker/inject/t0inject.sh"; then
    t0inject="/opt/injectworker/inject/t0inject.sh"
fi

#
# functions
#

modifykparams () {
#    echo     5 > /proc/sys/vm/dirty_background_ratio
#    echo    15 > /proc/sys/vm/dirty_ratio
    echo   256 > /proc/sys/vm/lower_zone_protection
    echo 16384 > /proc/sys/vm/min_free_kbytes
#    echo 1 > /proc/sys/fs/xfs/error_level
}

startcopyworker () {
    local local_file="/opt/copyworker/TransferSystem_Cessy.cfg"
    local reference_file="/nfshome0/smpro/configuration/TransferSystem_Cessy.cfg"

    if test -r "$reference_file"; then
        local local_time=`stat -t $local_file 2>/dev/null | cut -f13 -d' '`
        local reference_time=`stat -t $reference_file 2>/dev/null | cut -f13 -d' '`
        if test $reference_time -gt $local_time; then
            logger -s -t "SM INIT" "INFO: $reference_file is more recent than $local_file"
            logger -s -t "SM INIT" "INFO: I will overwrite the local configuration"
            mv $local_file $local_file.old.$local_time
            cp $reference_file $local_file
            sed -i "1i# File copied from $reference_file on `date`" $local_file
            chmod 644 $local_file
            chown cmsprod.root $local_file
        fi
    else
        logger -s -t "SM INIT" "WARNING: Can not read $reference_file"
    fi
    
    su - cmsprod -c "$t0control stop" >/dev/null 2>&1
    su - cmsprod -c "NCOPYWORKER=4 $t0control start"
}

startinjectworker () {
    local local_file="/opt/injectworker/.db.conf"
    local reference_file="/nfshome0/smpro/configuration/db.conf"

    if test -f "$reference_file"; then
        if test -f "$local_file"; then
            local local_time=`stat -t $local_file 2>/dev/null | cut -f13 -d' '`
            local reference_time=`stat -t $reference_file 2>/dev/null | cut -f13 -d' '`
            if test $reference_time -gt $local_time; then
                logger -s -t "SM INIT" "INFO: $reference_file is more recent than $local_file"
                logger -s -t "SM INIT" "INFO: I will overwrite the local configuration"
                mv $local_file $local_file.old.$local_time
                su - smpro -c "cp $reference_file $local_file"
                sed -i "1i# File copied from $reference_file on `date`" $local_file
                chmod 400 $local_file
                chown smpro.smpro $local_file
            fi
        else
            logger -s -t "SM INIT" "WARNING: $local_file doesn't exist, copying from $reference_file"
            su - smpro -c "cp $reference_file $local_file"
            sed -i "1i# File copied from $reference_file on `date`" $local_file
            chmod 400 $local_file
            chown smpro.smpro $local_file
        fi
    else
        logger -s -t "SM INIT" "WARNING: Can not read $reference_file"
    fi

    su - smpro -c "$t0inject stop" >/dev/null 2>&1
    rm -f /tmp/.20*-${hname}-*.log.lock
    su - smpro -c "$t0inject start"
}

startcopymanager () {
    local local_file="/opt/copymanager/TransferSystem_Cessy.cfg"
    local reference_file="/nfshome0/smpro/configuration/TransferSystem_Cessy.cfg"

    if test "$hname" != "$cmhost"; then
        echo "This host is not configured to by CopyManager: $hname != $cmhost"
        return;
    fi

    if test -r "$reference_file"; then
        local local_time=`stat -t $local_file 2>/dev/null | cut -f13 -d' '`
        local reference_time=`stat -t $reference_file 2>/dev/null | cut -f13 -d' '`
        if test $reference_time -gt $local_time; then
            logger -s -t "SM INIT" "INFO: $reference_file is more recent than $local_file"
            logger -s -t "SM INIT" "INFO: I will overwrite the local configuration"
            mv $local_file $local_file.old.$local_time
            cp $reference_file $local_file
            sed -i "1i# File copied from $reference_file on `date`" $local_file
            chmod 644 $local_file
            chown cmsprod.root $local_file
        fi
    else
        logger -s -t "SM INIT" "WARNING: Can not read $reference_file"
    fi
    
    su - cmsprod -c "$t0cmcontrol stop" >/dev/null 2>&1
    su - cmsprod -c "$t0cmcontrol start"
}

start () {
    case $hname in
        cmsdisk0)
            echo "cmsdisk0 needs manual treatment"
            return 0;
            ;;
        srv-S2C17-01)
            ;;
        srv-C2D05-02)
            for i in $store/satacmsdisk*; do 
                sn=`basename $i`
                if test -z "`mount | grep $sn`"; then
                    echo "Attempting to mount $i"
                    mount -L $sn $i
                fi
            done
            ;;
        srv-c2c07-* | srv-C2C07-* | srv-c2c06-* | srv-C2C06-*)

            if test -x "/sbin/multipath"; then
                echo "Refresh multipath devices"
                /sbin/multipath
            fi

            for i in $store/sata*a*v*; do 
                sn=`basename $i`
                if test -z "`mount | grep $sn`"; then
                    echo "Attempting to mount $i"
                    mount -L $sn $i
                fi
            done

            if test -x "/sbin/multipath"; then
                echo "Flushing unused multipath devices"
                /sbin/multipath -F
            fi

            modifykparams
            ;;
        *)
            echo "Unknown host: $hname"
            return 1;
            ;;
    esac

    if test -n "$SM_LA_NFS" -a "$SM_LA_NFS" != "local"; then
        if test -z "`mount | grep $lookarea`"; then
            mkdir -p $lookarea
            chmod 000 $lookarea
            echo "Attempting to mount $lookarea"
            mount -t nfs -o rsize=32768,wsize=32768,timeo=14,intr $SM_LA_NFS $lookarea
        fi
    fi

    if test -n "$SM_CALIB_NFS" -a -n "$SM_CALIBAREA"; then
        if test -z "`mount | grep $SM_CALIBAREA`"; then
            mkdir -p $SM_CALIBAREA
            chmod 000 $SM_CALIBAREA
            echo "Attempting to mount $SM_CALIBAREA"
            mount -t nfs -o rsize=32768,wsize=32768,timeo=14,intr $SM_CALIB_NFS $SM_CALIBAREA
        fi
    fi

    startcopymanager
    startcopyworker
    startinjectworker

    return 0;
}

stopcopyworker () {
    su - cmsprod -c "$t0control stop"

    counter=1;
    while [ $counter -le 10 ]; do
        teststr=done`ps ax | grep Copy | grep -v Copy`
        if test "$teststr" = "done"; then
            break;
        fi
        sleep 6;
        counter=`expr $counter + 1`;
    done

    killall -q rfcp
}

stopinjectworker () {
    su - smpro -c "$t0inject stop"
    rm -f /tmp/.20*-${hname}-*.log.lock
}

stopcopymanager () {
    if test "$hname" != "$cmhost"; then
        return;
    fi

    su - cmsprod -c "$t0cmcontrol stop"
}

stopworkers () {
    stopinjectworker
    stopcopyworker
    stopcopymanager
}

stop () {
    case $hname in
        cmsdisk0)
            echo "cmsdisk0 needs manual treatment"
            return 0;
            ;;
        srv-S2C17-01)
            stopworkers
            ;;
        srv-C2D05-02)
            stopworkers
            for i in $store/satacmsdisk*; do 
                sn=`basename $i`
                if test -n "`mount | grep $sn`"; then
                    echo "Attempting to unmount $i"
                    umount $i
                fi
            done
            ;;
        srv-c2c07-* | srv-C2C07-* | srv-c2c06-* | srv-C2C06-*)
            stopworkers
            for i in $store/sata*a*v*; do 
                sn=`basename $i`
                if test -n "`mount | grep $sn`"; then
                    echo "Attempting to unmount $i"
                    umount $i
                fi
            done
            ;;
        *)
            echo "Unknown host: $hname"
            return 1;
            ;;
    esac

    if test -n "$SM_LA_NFS" -a "$SM_LA_NFS" != "local"; then
        if test -n "`mount | grep $lookarea`"; then
            echo "Attempting to unmount $lookarea"
            umount -f $lookarea
        fi
    fi

    if test -n "$SM_CALIB_NFS" -a -n "$SM_CALIBAREA"; then
        if test -n "`mount | grep $SM_CALIBAREA`"; then
            echo "Attempting to umount $SM_CALIBAREA"
            umount -f $SM_CALIBAREA
        fi
    fi
    return 0;
}

printmstat () {
    if test -n "`mount | grep $2`"; then
        mounted=mounted
    else
        mounted=umounted
    fi
    echo "$1 $mounted"
}

status () {
    echo "*** $hname *** sm status"
    case $hname in
        cmsdisk0)
            echo "cmsdisk0 needs manual treatment"
            return 0;
            ;;
        srv-S2C17-01)
            ;;
        srv-C2D05-02)
            for i in $store/satacmsdisk*; do 
                sn=`basename $i`
                printmstat $i $sn
            done
            ;;
        srv-c2c07-* | srv-C2C07-* | srv-c2c06-* | srv-C2C06-*)
            for i in $store/sata*a*v*; do 
                sn=`basename $i`
                printmstat $i $sn
            done
            ;;
        *)
            echo "Unknown host: $hname"
            return 1;
            ;;
    esac

    if test -n "$SM_LA_NFS" -a "$SM_LA_NFS" != "local"; then
        printmstat $lookarea $lookarea
    fi

    if test -n "$SM_CALIB_NFS" -a -n "$SM_CALIBAREA"; then
        printmstat $SM_CALIBAREA $SM_CALIBAREA
    fi

    su - smpro -c "$t0inject status"
    su - cmsprod -c "$t0control status"
    su - cmsprod -c "$t0cmcontrol status"
    return 0
}

# See how we were called.
case "$1" in
    start)
	start
	RETVAL=$?
	;;
    stop)
	stop
	RETVAL=$?
	;;
    status)
	status
	RETVAL=$?
	;;
    startall)
	start
	RETVAL=$?
	;;
    stopall)
	stop
	RETVAL=$?
	;;
    statusall)
	status
	RETVAL=$?
	;;
    startinject)
	startinjectworker
	RETVAL=$?
	;;
    stopinject)
	stopinjectworker
	RETVAL=$?
	;;
    statusinject)
        su - smpro -c "$t0inject status"
	RETVAL=$?
	;;
    startcopy)
	startcopyworker
	RETVAL=$?
	;;
    stopcopy)
	stopcopyworker
	RETVAL=$?
	;;
    statuscopy)
        su - cmsprod -c "$t0control status"
	RETVAL=$?
	;;
    startmanager)
	startcopymanager
	RETVAL=$?
	;;
    stopmanager)
	stopcopymanager
	RETVAL=$?
	;;
    statusmanager)
        su - cmsprod -c "$t0cmcontrol status"
	RETVAL=$?
	;;
    *)
	echo $"Usage: $0 {start|stop|status|startinject|stopinject|statusinject|startcopy|stopcopy|statuscopy}"
	RETVAL=1
	;;
esac
exit $RETVAL
