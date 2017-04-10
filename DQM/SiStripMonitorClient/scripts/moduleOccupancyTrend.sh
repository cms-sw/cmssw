#!/bin/bash

if [[ "$#" == "0" ]]; then
    echo "usage: 'moduleOccupancyTrend.sh RootFileDirectory Dataset ModuleListFile UserCertFile UserKeyFile RunListFile'";
    exit 1;
fi

mkdir /data/users/tmpTrend
cp $6  /data/users/tmpTrend/.
cp $3  /data/users/tmpTrend/.

cd  /data/users/tmpTrend

rm -f modulefile

cat $6 | while read line
do

moduleOccupancyPlots.sh $1 $2 $line $3 $4 $5

done

ls *.root > rootfilestmp.txt

cat $3 | while read line
do
echo $line >> modulefile
done

echo "Summary" >> modulefile

moduleOccupancyTrend rootfilestmp.txt modulefile

rm modulefile

cd -

cp  /data/users/tmpTrend/* .
rm -rf  /data/users/tmpTrend
