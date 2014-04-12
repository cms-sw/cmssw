#!/bin/bash

for RUN in `cat RunList.txt`; do

#cp dbfile_31X_IdealConditions.db dbfile.db

IOVLIST=`echo $IOVLIST $RUN`

DATASET=`echo $RUN | cut -c 1-3`
AFSPATH=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/Express/$DATASET
DQMFILE=DQM_V0001_R000${RUN}__ExpressMuon__CRAFT09-Express-v1__FEVT.root

echo "Analyzing Run $RUN ..."
echo "Using Dataset: $DATASET"
echo "Using DQMFile: $DQMFILE"

echo "Creating BadAPVIdentifier config from template"
cat template_SiStripQualityBadAPVIdentifierRoot_cfg.py |sed -e "s@insertRun@$RUN@g" -e "s@insertCastorPath@$AFSPATH@" -e "s@insertDataset@$DATASET@" -e "s@insertDQMFile@$DQMFILE@" > SiStripQualityBadAPVIdentifierRoot_cfg.py

echo "Starting cmsRun BadAPVIdentifier"
cmsRun SiStripQualityBadAPVIdentifierRoot_cfg.py

echo "Creating HotStripIdentification config from template"
cat template_SiStripQualityHotStripIdentifierRoot_cfg.py |sed -e "s@insertRun@$RUN@g" -e "s@insertCastorPath@$AFSPATH@" -e "s@insertDataset@$DATASET@" -e "s@insertDQMFile@$DQMFILE@" > SiStripQualityHotStripIdentifierRoot_cfg.py

echo "Starting cmsRun HotStripIdentification"
cmsRun SiStripQualityHotStripIdentifierRoot_cfg.py

echo "Creating Merge config from template"
cat template_SiStripBadComponents_merge_cfg.py |sed -e "s@insertRun@$RUN@g" > SiStripBadComponents_merge_cfg.py

echo "Starting cmsRun Merge"
cmsRun SiStripBadComponents_merge_cfg.py

echo "Creating SiStripQualityStatistics_offline config from template"
cat template_SiStripQualityStatistics_offline_cfg.py |sed -e "s@insertRun@$RUN@" > SiStripQualityStatistics_offline_cfg.py

echo "Starting cmsRun SiStripQualityStatistics_offline"
cmsRun SiStripQualityStatistics_offline_cfg.py > out.tmp

cat out.tmp | awk 'BEGIN{doprint=0}{if(match($0,"New IOV")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;print "";} if(doprint==1) print $0}' > BadComponents_$RUN.txt

rm out.tmp

echo "Creating the directories on cmstac12 in /data/local1/cmstkmtc/CRAFTReproIn31X/output/BadStripAnalysis/GR09/$RUN"
mkdir /data/local1/cmstkmtc/CRAFTReproIn31X/output/BadStripAnalysis/GR09/$RUN
mkdir /data/local1/cmstkmtc/CRAFTReproIn31X/output/BadStripAnalysis/GR09/$RUN/ExpressTightlyFiltered
mkdir /data/local1/cmstkmtc/CRAFTReproIn31X/output/BadStripAnalysis/GR09/$RUN/ExpressTightlyFiltered/TkMap

echo "Moving the output files to the proper directories"
cp dbfile.db /data/local1/cmstkmtc/CRAFTReproIn31X/output/BadStripAnalysis/GR09/$RUN/ExpressTightlyFiltered
mv BadAPVOccupancy_${RUN}.root BadComponents_${RUN}.txt HotStripsOccupancy_${RUN}.root /data/local1/cmstkmtc/CRAFTReproIn31X/output/BadStripAnalysis/GR09/$RUN/ExpressTightlyFiltered
mv TkMapBadComponents_offline* /data/local1/cmstkmtc/CRAFTReproIn31X/output/BadStripAnalysis/GR09/$RUN/ExpressTightlyFiltered/TkMap

echo "Run $RUN finished"

done;

echo "Preparing the sqlite and metadata files"
FIRSTRUN=`cat RunList.txt | awk '{if(NR==1) print $1}'`

ID1=`uuidgen -t`
cp dbfile.db SiStripHotAPVs@${ID1}.db
cat template_SiStripHotAPVs.txt | sed -e "s@insertFirstRun@$FIRSTRUN@g" -e "s@insertIOV@$IOVLIST@" > SiStripHotAPVs@${ID1}.txt

ID2=`uuidgen -t`
cp dbfile.db SiStripHotStrips@${ID2}.db
cat template_SiStripHotStrips.txt | sed -e "s@insertFirstRun@$FIRSTRUN@g" -e "s@insertIOV@$IOVLIST@" > SiStripHotStrips@${ID2}.txt

ID3=`uuidgen -t`
cp dbfile.db SiStripHotComponents_merged@${ID3}.db
cat template_SiStripHotComponents_merged.txt | sed -e "s@insertFirstRun@$FIRSTRUN@g" -e "s@insertIOV@$IOVLIST@" > SiStripHotComponents_merged@${ID3}.txt

echo "-----------------------------------"
echo "Sqlite and metadata files are ready"
echo "1st Sqlite: SiStripHotAPVs@${ID1}.db, 1st Metadata: SiStripHotAPVs@${ID1}.txt"
echo "2nd Sqlite: SiStripHotStrips@${ID2}.db, 2nd Metadata: SiStripHotStrips@${ID2}.txt"
echo "3rd Sqlite: SiStripHotComponents_merged@${ID3}.db, 3rd Metadata: SiStripHotComponents_merged@${ID3}.txt"
echo "-----------------------------------"
echo "Now the sqlite and corresponding metadata files have to be moved to the Popcon dropbox!"
echo "Do: scp <sqlite-file> <metadata-file> <username>@cmsusr5:/nfshome0/popcondev/SiStripJob"
