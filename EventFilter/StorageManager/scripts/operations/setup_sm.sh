#!/bin/sh
# $Id: setup_sm.sh,v 1.3 2008/04/01 16:56:59 loizides Exp $

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
	    mount -L `basename $i` $i
	done
        mkdir -p $lookarea
        chmod 000 $lookarea
        if test -n "$SM_LA_NFS"; then
            mount -t nfs -o rsize=32768,wsize=32768,timeo=14,intr $SM_LA_NFS $lookarea
        fi
	nname="node"`echo $hname | cut -d- -f3` 
	su - cmsprod -c "~cmsprod/$nname/t0_control.sh start"
	su - smpro -c "~smpro/scripts/t0inject.sh start"
	;;
    *)
        echo "Unknown host: $hname"
        ;;
esac
