#!/bin/sh
# $Id: setup_sm.sh,v 1.4 2008/04/28 13:34:22 loizides Exp $

if test -e "/etc/profile.d/sm_env.sh"; then 
    source /etc/profile.d/sm_env.sh;
fi

store=/store
if test -n "$SM_STORE"; then
    store=$SM_STORE
fi
lookarea=$store/lookarea
if test -n "$SM_LOOKAREA"; then
    lookarea=$SM_LOOKAREA
fi

hname=`hostname | cut -d. -f1`;
case $hname in
    cmsdisk0)
        ;;
    cmsdisk1)
        ;;
    srv-*)
	for i in $store/sata*a*v*; do 
            sn=`basename $i`
            if test -z "`mount | grep $sn`"; then
                mount -L $sn $i
            fi
	done
        if test -n "$SM_LA_NFS"; then
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
	nname="node"`echo $hname | cut -d- -f3` 
	su - cmsprod -c "~cmsprod/$nname/t0_control.sh stop" >/dev/null 2>&1
	su - cmsprod -c "~cmsprod/$nname/t0_control.sh start"
	su - smpro -c "~smpro/scripts/t0inject.sh stop" >/dev/null 2>&1
	su - smpro -c "~smpro/scripts/t0inject.sh start"
	;;
    *)
        echo "Unknown host: $hname"
        ;;
esac
