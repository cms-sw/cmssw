#!/bin/bash
# $Id:$

# run this script to get a mapping between hostname and wwpn
# the mapping has the syntax of a perl hash and can be put
# into the mk_satamap.pl script.

for i in 13 14 15 16 17 18 19 20; do
    ping -c1 -q -W3 srv-c2c07-$i > /dev/null
    if test "$?" = "0"; then
        res=`ssh srv-c2c07-$i grep scsi-qla0-adapter-port /proc/scsi/qla2xxx/* 2>/dev/null | cut -d= -f2 | cut -d\; -f1`;
        re2=`echo $res | tr 'a-z' 'A-Z'`;
        re3=`echo $re2 | cut -b1-2`-`echo $re2 | cut -b3-4`-`echo $re2 | cut -b5-6`-`echo $re2 | cut -b7-8`-`echo $re2 | cut -b9-10`-`echo $re2 | cut -b11-12`-`echo $re2 | cut -b13-14`-`echo $re2 | cut -b15-16`
        echo -e \"srv-c2c07-$i\" =\> \"$re3\",
    else
        echo -e \"srv-c2c07-$i\" =\> \"unknown\",
    fi
done
