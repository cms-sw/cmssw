#!/bin/bash
# $Id: setup_sm.sh,v 1.2 2008/04/01 16:52:57 loizides Exp $

hname=`hostname | cut -d. -f1`;

case $hname in
#to be done
#    cmsdisk0)
#  ;;
    *)
	for i in /store/sata*a*v*; do 
	    mount -L `basename $i` $i
	done
	test -e /store/lookarea && ln -fs /store/lookarea /lookarea
	mount cmsmon:/cms/mon/data/lookarea_SM  /store/lookarea
	nname="node"`echo $hname | cut -d- -f3` 
	su - cmsprod -c "~cmsprod/$nname/t0_control.sh start"
	;;
esac
