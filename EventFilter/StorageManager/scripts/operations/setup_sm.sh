#!/bin/sh
# $Id: setup_sm.sh,v 1.27 2008/11/14 07:02:16 loizides Exp $

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
    srv-c2c06-* | srv-C2C06-*)
        nname="nottobeused"
        ;;
    *)
        ;;
esac

t0control="~cmsprod/$nname/t0_control.sh";
if test -e "/opt/copyworker/t0_control.sh"; then
    t0control="/opt/copyworker/t0_control.sh"
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

stopunwantedservices () {
    /etc/init.d/cups     stop >/dev/null 2>&1
    /etc/init.d/squid    stop >/dev/null 2>&1
    /etc/init.d/xfs      stop >/dev/null 2>&1
    /etc/init.d/sendmail stop >/dev/null 2>&1
    /etc/init.d/gpm      stop >/dev/null 2>&1
}

startwantedservices () {
    ms="~smpro/sm_scripts_cvs/operations/monitoringSar.sh";
    if test -e $ms; then 
        $ms >> /var/log/monitoringSar.log &
    fi
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
            stopunwantedservices

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

            startwantedservices
            modifykparams
            #mount -oro,remount /dev/sda1 /boot/
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

    su - cmsprod -c "$t0control stop" >/dev/null 2>&1
    su - cmsprod -c "NCOPYWORKER=2 $t0control start"
    su - smpro -c "$t0inject stop" >/dev/null 2>&1
    rm -f /tmp/.20*-${hname}-*.log.lock
    su - smpro -c "$t0inject start"

    return 0;
}

stopworkers () {
    su - smpro -c "$t0inject stop"
    rm -f /tmp/.20*-${hname}-*.log.lock
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
            killall -5 monitoringSar.sh
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
    *)
	echo $"Usage: $0 {start|stop|status}"
	RETVAL=1
	;;
esac
exit $RETVAL
