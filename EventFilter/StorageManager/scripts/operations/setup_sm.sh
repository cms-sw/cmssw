#!/bin/sh
# $Id: setup_sm.sh,v 1.56 2012/07/10 15:53:49 babar Exp $

if test -e "/etc/profile.d/sm_env.sh"; then 
    source /etc/profile.d/sm_env.sh
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

cmhost=srv-C2C06-20

hname=`hostname | cut -d. -f1`

t0control="~cmsprod/TransferTest/T0/t0_control.sh"
if test -e "/opt/copyworker/t0_control.sh"; then
    t0control="/opt/copyworker/t0_control.sh"
fi

t0cmcontrol="~cmsprod/TransferTest/old_t0_transferstatusworker.sh_old"
if test -e "/opt/copymanager/t0_control.sh"; then
    t0cmcontrol="/opt/copymanager/t0_control.sh"
fi

t0inject="~smpro/scripts/t0inject.sh"
if test -e "/opt/injectworker/inject/t0inject.sh"; then
    t0inject="/opt/injectworker/inject/t0inject.sh"
fi

t0notify="~smpro/scripts/t0notify.sh"
if test -e "/opt/injectworker/inject/t0notify.sh"; then
    t0notify="/opt/injectworker/inject/t0notify.sh"
fi

# For xfs_admin
PATH=$PATH:/usr/sbin

#
# functions
#

# Checks if a host is running the currently supported SLC version
# to prevent starting up things if not, to avoid crashing the CopyManager
checkSLCversion () {
    slc_release=$(cat /etc/redhat-release)
    case $slc_release in
        *5.3*)
            ;;
        *4.4*)
            echo "This host is running $slc_release, which is NO LONGER compatible with the current SLC5 CopyManager" >&2
            echo "Therefore nothing will be started, but the disks will be mounted" >&2
            exit
            ;;
        *)
            echo "This host is running $slc_release, which is UNKOWN" >&2
            echo "Therefore nothing will be started, but the disks will be mounted" >&2
            exit
            ;;
    esac
}

# Mounts a disk, looking for its label
mountByLabel () {
    if [ ! -d $1 ]; then
        echo "$1 is not a directory, not trying to mount it!"
        return 1
    fi
    sn=`basename $1`
    # First, trying to mount by label
    mount -L $sn $1 >/dev/null
    # If the previous mount failed, it's most likely because it found a disk
    # inside the multipath, instead of the multipath one, so try manually
    if mount | grep -q $sn; then
        echo "Mounted $1 using label"
    else
        device=''
        for dev in /dev/mapper/mpath*; do
            (xfs_admin -l $dev || e2label $dev) | grep -q 'label = "'$sn'"' && device=$dev
        done
        if [ -b "$device" ]; then
            echo "Attempting to mount $1 from $device"
            if mount $device $1; then
                echo "Mounted $1 manually from $device"
            else
                echo "Could not mount $1 from $device!"
            fi
        else
            echo "Could not mount $1: no device in /dev/mapper/mpath* with label $sn"
        fi
    fi
}

modifykparams () {
#    echo     5 > /proc/sys/vm/dirty_background_ratio
#    echo    15 > /proc/sys/vm/dirty_ratio
    echo   256 > /proc/sys/vm/lower_zone_protection
    echo 16384 > /proc/sys/vm/min_free_kbytes
#    echo 1 > /proc/sys/fs/xfs/error_level
}

startcopyworker () {
    checkSLCversion
    local local_file="/opt/copyworker/TransferSystem_Cessy.cfg"
    local reference_file="/nfshome0/smpro/configuration/TransferSystem_Cessy.cfg"

    mkdir -p /store/copyworker
    chmod 755 /store/copyworker
    chown 5410:0 /store/copyworker

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
    su - cmsprod -c "$t0control start"
}

startinjectworker () {
    checkSLCversion
    local local_file="/opt/injectworker/.db.conf"
    local reference_file="/nfshome0/smpro/configuration/db.conf"

    mkdir -p /store/injectworker
    chmod 755 /store/injectworker
    chown 40491:0 /store/injectworker 

    if test -f "$reference_file"; then
        if test -s "$local_file"; then
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
            if [ -f "$local_file" ]; then
                logger -s -t "SM INIT" "WARNING: $local_file is empty, copying from $reference_file"
                rm -f "$local_file"
            else
                logger -s -t "SM INIT" "WARNING: $local_file doesn't exist, copying from $reference_file"
            fi
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

startnotifyworker () {
    checkSLCversion
    local local_file="/opt/injectworker/.db.conf"
    local reference_file="/nfshome0/smpro/configuration/db.conf"

    if test -f "$reference_file"; then
        if test -s "$local_file"; then
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
            if [ -f "$local_file" ]; then
                logger -s -t "SM INIT" "WARNING: $local_file is empty, copying from $reference_file"
                rm -f "$local_file"
            else
                logger -s -t "SM INIT" "WARNING: $local_file doesn't exist, copying from $reference_file"
            fi
            su - smpro -c "cp $reference_file $local_file"
            sed -i "1i# File copied from $reference_file on `date`" $local_file
            chmod 400 $local_file
            chown smpro.smpro $local_file
        fi
    else
        logger -s -t "SM INIT" "WARNING: Can not read $reference_file"
    fi

    su - smpro -c "$t0notify stop" >/dev/null 2>&1
    su - smpro -c "$t0notify start"
}

startcopymanager () {
    checkSLCversion
    local local_file="/opt/copymanager/TransferSystem_Cessy.cfg"
    local reference_file="/nfshome0/smpro/configuration/TransferSystem_Cessy.cfg"

    if test "$hname" != "$cmhost"; then
        echo "This host is not configured to be THE CopyManager: $hname != $cmhost"
        return
    fi

    mkdir -p /store/copymanager
    chmod 755 /store/copymanager
    chown 5410:0 /store/copymanager

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
    nbmount=0
    case $hname in
        cmsdisk0)
            echo "cmsdisk0 needs manual treatment"
            return 0
            ;;
        srv-S2C17-01)
            ;;
        srv-C2D05-02)
            nbmount=1
            ;;
        dvsrv-b1a02-30-01)
            nbmount=3
            ;;
        srv-c2c07-* | srv-C2C07-* | srv-c2c06-* | srv-C2C06-* | dvsrv-C2F37-*)
            nbmount=4
            ;;
        *)
            echo "Unknown host: $hname"
            return 1
            ;;
    esac

    if test -x "/sbin/multipath"; then
        echo "Refresh multipath devices"
        /sbin/multipath
    fi

    mounts=(/store/sata*)
    if [ "$mounts" = "/store/sata*" ]; then
        echo "No mountpoint define, assuming install needed, calling make_all.sh"
        ( cd ~smpro/sm_scripts_cvs/operations/ && ./makeall.sh )
        mounts=(/store/sata*)
    fi

    case ${#mounts[@]} in
    $nbmount)   # We have as much as we should
        for i in ${mounts[@]}; do
            mountByLabel $i
        done
    ;;
    *)
        echo -e "\e[33mWARNING: Odd number of mount points: ${#mounts[@]} (should be $nbmount). Skipping mounts\e[0m"
    ;;
    esac

    if test -x "/sbin/multipath"; then
        echo "Flushing unused multipath devices"
        /sbin/multipath -F
    fi

    modifykparams

    if test -n "$SM_LA_NFS" -a "$SM_LA_NFS" != "local"; then
        if test -z "`mount | grep $lookarea`"; then
            mkdir -p $lookarea
            chmod 000 $lookarea
            echo "Attempting to mount $lookarea"
            mount -t nfs -o rsize=32768,wsize=32768,timeo=14,intr,bg $SM_LA_NFS $lookarea
        fi
    fi

    if test -n "$SM_CALIB_NFS" -a -n "$SM_CALIBAREA"; then
        if test -z "`mount | grep $SM_CALIBAREA`"; then
            mkdir -p $SM_CALIBAREA
            chmod 000 $SM_CALIBAREA
            echo "Attempting to mount $SM_CALIBAREA"
            mount -t nfs -o rsize=32768,wsize=32768,timeo=14,intr,bg $SM_CALIB_NFS $SM_CALIBAREA
        fi
    fi

    /sbin/service copymanager start
    /sbin/service copyworker start
    /sbin/service injectworker start
    /sbin/service notifyworker start

    return 0
}

stopcopyworker () {
    su - cmsprod -c "$t0control stop"

    counter=1
    while [ $counter -le 10 ]; do
        if pgrep -u cmsprod CopyWorker.pl >/dev/null; then
            echo -n .
            sleep 2
        else
            break
        fi
        counter=`expr $counter + 1`
    done
    if [ $counter -ge 10 ]; then
        pkill -9 -u cmsprod CopyWorker.pl
        pkill    -u cmsprod rfcp
        echo -n 'Killed! '
    fi
    return 0
}

stopinjectworker () {
    su - smpro -c "$t0inject stop"
    rm -f /tmp/.20*-${hname}-*.log.lock
}

stopnotifyworker () {
    su - smpro -c "$t0notify stop"
}

stopcopymanager () {
    if test "$hname" != "$cmhost"; then
        return
    fi

    su - cmsprod -c "$t0cmcontrol stop"
}

stopworkers () {
    /sbin/service notifyworker stop
    /sbin/service injectworker stop
    /sbin/service copyworker stop
    /sbin/service copymanager stop
}

stop () {
    case $hname in
        cmsdisk0)
            echo "cmsdisk0 needs manual treatment"
            return 0
            ;;
        srv-S2C17-01)
            stopworkers
            ;;
        srv-C2D05-02 | dvsrv-b1a02-30-01)
            stopworkers
            for i in $store/satacmsdisk*; do 
                sn=`basename $i`
                if test -n "`mount | grep $sn`"; then
                    echo "Attempting to unmount $i"
                    umount $i
                fi
            done
            ;;
        srv-c2c07-* | srv-C2C07-* | srv-c2c06-* | srv-C2C06-* | dvsrv-C2F37-*)
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
            return 1
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
    return 0
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
            return 0
            ;;
        srv-S2C17-01)
            ;;
        srv-C2D05-02 | dvsrv-b1a02-30-01)
            for i in $store/satacmsdisk*; do 
                sn=`basename $i`
                printmstat $i $sn
            done
            ;;
        srv-c2c07-* | srv-C2C07-* | srv-c2c06-* | srv-C2C06-* | dvsrv-C2F37-*)
            for i in $store/sata*a*v*; do 
                sn=`basename $i`
                printmstat $i $sn
            done
            ;;
        *)
            echo "Unknown host: $hname"
            return 1
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
    startnotify)
        startnotifyworker
        RETVAL=$?
        ;;
    stopnotify)
        stopnotifyworker
        RETVAL=$?
        ;;
    statusnotify)
        su - smpro -c "$t0notify status"
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
        echo $"Usage: $0
        {start|stop|status|startinject|stopinject|statusinject|startnotify|stopnotify|statusnotify|startcopy|stopcopy|statuscopy|startmanager|stopmanager|statusmanager}"
        RETVAL=1
        ;;
esac
exit $RETVAL
