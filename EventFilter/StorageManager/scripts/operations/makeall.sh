#!/bin/bash
# $Id: makeall.sh,v 1.2 2008/06/11 13:58:04 loizides Exp $ 

hname=`hostname | cut -d. -f1`
echo "Host I am running on: $hname"

exec 10< satamap.txt

while read LINE <&10; do

    echo Parsing $LINE
    sata=`echo $LINE | cut -d" " -f1 | cut -d"a" -f3`
    arr=` echo $LINE | cut -d" " -f2 | cut -d"a" -f2`
    vol=` echo $LINE | cut -d" " -f3 | cut -d"v" -f2`
    id=`  echo $LINE | cut -d" " -f4`
    host=`echo $LINE | cut -d" " -f5`

    if [ $host = $hname ] ; then
	echo "Working on: "
        echo "Sata   $sata"
        echo "Array  $arr"
        echo "Volume $vol"
        echo "Id     $id"
        echo "Host   $host"
        dpath=/dev/mapper/`/sbin/multipath -l | grep -i $id | cut -d" " -f1`
        echo "Path   $dpath"
        echo "mkvol.sh $id $sata $arr $vol"
        ./mkvol.sh $id $sata $arr $vol
    fi
done

./mkstore.sh
./setup_sm.sh
exec 10>&-
