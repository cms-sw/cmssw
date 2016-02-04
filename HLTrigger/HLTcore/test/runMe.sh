#!/bin/tcsh

# usage: ./runMe.sh <HLT Key Name>

cmsenv
rehash

echo "HLT Key: " $1

# remove old temporary files if any
touch   ../python/tmpPrescaleService.py
/bin/rm ../python/tmpPrescaleService.py*

# extract PrescaleService using provided HLT Key
edmConfigFromDB --nopsets --noes --nopaths --cff --services PrescaleService --configName $1 > ../python/tmpPrescaleService.py

touch   put.py  get.py
/bin/rm put.py* get.py*
edmConfigDump ./putHLTPrescaleTable.py > put.py
edmConfigDump ./getHLTPrescaleTable.py > get.py
/bin/rm ../python/tmpPrescaleService.py*

echo
echo "cmsRun to put it into the DB"
cmsRun put.py

echo
echo "Dump content of DB"
foreach tag (`cmscond_list_iov -c sqlite_file:HLTPrescaleTable.db -a`)
    echo $tag
    cmscond_list_iov -c sqlite_file:HLTPrescaleTable.db -t $tag
end

echo
echo "cmsRun to get it from the DB"
cmsRun get.py
