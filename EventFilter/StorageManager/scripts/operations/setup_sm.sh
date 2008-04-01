#!/bin/bash
#$Id:$

hname=`hostname | cut -d. -f1`;

case $hname in
srv-C2C07-19)
  mount -L6EE99434 /store/sata03ar01vol01
  mount -L6EE99453 /store/sata03ar01vol02
  mount -L6EE99463 /store/sata03ar01vol03
  mount cmsmon:/cms/mon/data/lookarea_SM  /store/lookarea
  su - cmsprod -c "~cmsprod/node19/t0_control.sh start"
  ;;
srv-C2C07-20)
  mount -L6EE993A7 /store/sata03ar02vol01
  mount -L6EE993BA /store/sata03ar02vol02
  mount -L6EE993CD /store/sata03ar02vol03
  mount cmsmon:/cms/mon/data/lookarea_SM  /store/lookarea
  su - cmsprod -c "~cmsprod/node20/t0_control.sh start"
  ;;
*)
  Message="Hostname not known to setup.sh"
  ;;
esac
