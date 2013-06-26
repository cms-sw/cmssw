#!/bin/bash

# This script can be used to append existing tags on orcon to new tags.
# It downloads tags from orcoff to the dbfile.db.
# It then prepares the files to append the orcoff tags to the sqlite ones.
# It runs load_iov to create the new tables.

tagIdentifier="GR09"
oldVersion=31X_v1
newVersion=ideal
newIOV=100355

for tag in `cmscond_list_iov -c oracle://cms_orcoff_prod/CMS_COND_21X_STRIP -P /afs/cern.ch/cms/DB/conddb | grep ${tagIdentifier}`; do
    echo $tag
    cmscond_export_iov -D CondFormatsSiStripObjects -P /afs/cern.ch/cms/DB/conddb -s  oracle://cms_orcoff_prod/CMS_COND_21X_STRIP -d sqlite_file:dbfile.db -i $tag -t $tag
done

for tag in `cmscond_list_iov -c sqlite_file:dbfile.db -P /afs/cern.ch/cms/DB/conddb | grep ${tagIdentifier}_${oldVersion}`; do
    # # First the ideal tag which will start from iov = 1
    idealTag=`echo $tag | sed -e "s/${oldVersion}/${newVersion}/g"`
    echo "Dumping payloadToken for ${idealTag}"
    cmscond_list_iov -c sqlite_file:dbfile.db -t $idealTag > ${tag}.txt
    echo "Dumping payloadToken for ${tag}"
    cmscond_list_iov -c sqlite_file:dbfile.db -t $tag >> ${tag}.txt

    # Remove five lines starting from line 6 (included)
    # -i means that it will modify directly the file
    sed -i '6, +4 d' ${tag}.txt
    oldIOV=$[${newIOV}-1]
    echo "Ideal IOV is from 1 to $oldIOV"
    echo "o2o IOV is from $newIOV to infinity"
    # Replace only in line 5
    sed -i "5s/4294967295/${oldIOV}/" ${tag}.txt
    sed -i "6s/1 /${newIOV}/" ${tag}.txt
    sed -i "1s/_hlt//" ${tag}.txt

    cmscond_load_iov -c sqlite_file:dbfile.db ${tag}.txt
done
