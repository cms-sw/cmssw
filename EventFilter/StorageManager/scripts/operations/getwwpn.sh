#!/bin/bash
# $Id: getwwpn.sh,v 1.2 2008/09/13 01:06:45 loizides Exp $
#
# run this script to get a mapping between hostname and wwpn
# the mapping has the syntax of a perl hash and can be put
# into the mk_satamap.pl script.
#

nodes=$1
if test -z $1; then 
    if test -e host_list.cfg; then
        nodes=`cat host_list.cfg`
    else
        nodes="";
        for i in `seq 13 20`; do
            nodes="$nodes srv-C2C07-$i"
        done
    fi
else
    nodes=`echo $nodes | tr c C`;
fi

for i in $nodes; do
    ping -c1 -q -W3 $i > /dev/null
    if test "$?" = "0"; then
        res=`ssh $i grep scsi-qla0-adapter-port /proc/scsi/qla2xxx/* 2>/dev/null | cut -d= -f2 | cut -d\; -f1`;
        re2=`echo $res | tr 'a-z' 'A-Z'`;
        re3=`echo $re2 | cut -b1-2`-`echo $re2 | cut -b3-4`-`echo $re2 | cut -b5-6`-`echo $re2 | cut -b7-8`-`echo $re2 | cut -b9-10`-`echo $re2 | cut -b11-12`-`echo $re2 | cut -b13-14`-`echo $re2 | cut -b15-16`
        echo -e \"$i\" =\> \"$re3\",
    else
        echo -e \"$i\" =\> \"unknown\",
    fi
done
