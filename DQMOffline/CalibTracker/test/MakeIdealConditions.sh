#!/bin/bash

if [ $# -ne 2 ]; then
    echo "You have to provide a <DestinationTag> and the <RunNumber> !!!"
    echo "E.g.: ./MakeIdealConditions.sh SiStripBadComponents_OfflineAnalysis_GR09_31X_v1_offline 110000"
    exit
fi

echo "Preparing the sqlite and metadata file"

TAG=$1
FIRSTRUN=$2

ID1=`uuidgen -t`
cp dbfile_31X_IdealConditions.db SiStripIdealConditions@${ID1}.db
cat template_SiStripIdealConditions.txt | sed -e "s@insertDestinationTag@$TAG@" -e "s@insertFirstRun@$FIRSTRUN@g" > SiStripIdealConditions@${ID1}.txt

echo "-----------------------------------"
echo "Sqlite and metadata files are ready"
echo "Sqlite file: SiStripIdealConditions@${ID1}.db, Metadata file: SiStripIdealConditions@${ID1}.txt"
echo "-----------------------------------"
echo "Now the sqlite and the corresponding metadata file have to be moved to the Popcon dropbox!"
echo "Do: scp <sqlite-file> <metadata-file> <username>@cmsusr5:/nfshome0/popcondev/SiStripJob"
