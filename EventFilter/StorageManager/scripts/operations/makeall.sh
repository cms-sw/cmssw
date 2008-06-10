#!/bin/bash
# 

hname=`hostname | cut -d. -f1`
#hname="srv-C2C07-15"
echo $hname

exec 10< satamap.txt

while read LINE <&10; do

echo $LINE

sata=`echo $LINE | cut -d" " -f1 | cut -d"a" -f3`
arr=` echo $LINE | cut -d" " -f2 | cut -d"a" -f2`
vol=` echo $LINE | cut -d" " -f3 | cut -d"v" -f2`
id=`  echo $LINE | cut -d" " -f4`
host=`echo $LINE | cut -d" " -f5`

if [ $host = $hname ] ; then

echo $sata
echo $arr
echo $vol
echo $id
echo $host


dpath=/dev/mapper/`/sbin/multipath -l | grep -i $id | cut -d" " -f1`
echo $dpath


echo "mkvol.sh $id $sata $arr $vol"


./mkvol.sh $id $sata $arr $vol


./mkstore.sh

./setup_sm.sh



fi

done

exec 10>&-
