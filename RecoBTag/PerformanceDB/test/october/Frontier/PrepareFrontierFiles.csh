#!/bin/tcsh

#
# please make sure you have the HEAD of CondTools/OracleDBA/test in the project area
#

#
# first, create the DB files
#
rm -f tables
cmscond_list_iov -a -c sqlite_file:../PhysicsPerformance.db> tables

rm -rf results
mkdir results
cd results
foreach i (`cat ../tables`)
    echo Working on $i
    rm -f ${i}.db
    cmscond_export_iov -D CondFormatsPhysicsToolsObjects -t $i -s sqlite_file:../../PhysicsPerformance.db -d sqlite_file:${i}.db  -l sqlite_file:LogExport.db
    rm -f text{$i}.txt
    cat ../template.txt | sed "s/TEMPLATE/$i/g" > text${i}.txt
    ../../../../../../CondTools/OracleDBA/test/renameFiles.sh ${i}.db text${i}.txt $i
rm -f text{$i}.txt ${i}.db
end

#
