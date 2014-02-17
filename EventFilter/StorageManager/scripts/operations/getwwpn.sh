#!/bin/bash
# $Id: getwwpn.sh,v 1.5 2009/08/17 13:21:20 gbauer Exp $
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
        for i in `seq 12 20`; do
            nodes="$nodes srv-C2C06-$i"
        done
        for i in `seq 12 20`; do
            nodes="$nodes srv-C2C07-$i"
        done
	nodes="$nodes dvsrv-C2f37-01"
	nodes="$nodes dvsrv-C2f37-02"
    fi
else
    nodes=`echo $nodes | tr c C`;
fi

for i in $nodes; do
    ping -c1 -q -W3 $i > /dev/null
    if test "$?" = "0"; then
        res=`ssh $i egrep "scsi-qla.-adapter-port" /proc/scsi/qla2xxx/\* 2>/dev/null | cut -d= -f2 | cut -d\; -f1`;
        re2=`echo $res | tr 'a-z' 'A-Z'`;
        re2a=`echo $re2 | cut -d" " -f1`;
        re2b=`echo $re2 | cut -d" " -f2`;

        re3a=`echo $re2a | cut -b1-2`-`echo $re2a | cut -b3-4`-`echo $re2a | cut -b5-6`-`echo $re2a | cut -b7-8`-`echo $re2a | cut -b9-10`-`echo $re2a | cut -b11-12`-`echo $re2a | cut -b13-14`-`echo $re2a | cut -b15-16`
        re3b=`echo $re2b | cut -b1-2`-`echo $re2b | cut -b3-4`-`echo $re2b | cut -b5-6`-`echo $re2b | cut -b7-8`-`echo $re2b | cut -b9-10`-`echo $re2b | cut -b11-12`-`echo $re2b | cut -b13-14`-`echo $re2b | cut -b15-16`


        echo -e \"$i\" =\> \"$re3a \ \ \ \    $re3b\",

    else
        echo -e \"$i\" =\> \"unknown\",
    fi
done
 
