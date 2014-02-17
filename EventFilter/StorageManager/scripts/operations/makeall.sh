#!/bin/bash
# $Id: makeall.sh,v 1.6 2010/01/29 16:15:35 babar Exp $ 

## *** added security feature: MUST give argument "makefilesys"
## or else file system will NOT be re-made ***

nofs=$1

if [ $nofs = "makefilesys" ] ; then
    echo " **** You have asked for the file system to be re-made.."
    echo " ...destroying all files on the SataBeast!...."
    sleep 3;
fi

hname=`hostname | cut -d. -f1`
hname=`echo $hname | tr 'a-z' 'A-Z'`;

echo "Host I am running on: $hname"

exec 10< ~smpro/configuration/script-generated/satamap.txt

while read LINE <&10; do

    echo Parsing $LINE
    sata=`echo $LINE | cut -d" " -f1 | cut -d"a" -f3`
    arr=` echo $LINE | cut -d" " -f2 | cut -d"a" -f2`
    vol=` echo $LINE | cut -d" " -f3 | cut -d"v" -f2`
    id=`  echo $LINE | cut -d" " -f4`
    host=`echo $LINE | cut -d" " -f5`


 host=`echo $host | tr 'a-z' 'A-Z'`;


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
        ./mkvol.sh $id $sata $arr $vol $nofs
    fi
done

echo "RUNNING mkstore...."
./mkstore.sh
# Commenting this out as it might loop otherwise - Babar - 01/29/2010
#echo "RUNNING setup_sm...."
#./setup_sm.sh
exec 10>&-
