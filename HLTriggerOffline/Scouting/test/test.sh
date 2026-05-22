#!/bin/bash -ex

LOCALPATH='/eos/cms/store/group/tsg-phase2/user/jprendi/NERD25/MoreStats/Muons/HLT/'
echo "Input source: |${LOCALPATH}|"
LOCALFILES=$(ls -1 ${LOCALPATH} | grep Scouting)
ALL_FILES=""
for f in ${LOCALFILES[@]}; do
    ALL_FILES+="file:${LOCALPATH}/${f},"
done

# Remove the last character
ALL_FILES="${ALL_FILES%?}"
echo "Discovered files: $ALL_FILES"

cmsDriver.py step2 -s DQM:hltScoutingTrackMonitor \
             --conditions 160X_dataRun3_HLT_v1 \
             --datatier DQMIO \
             -n 10000 \
             --eventcontent DQMIO \
             --geometry DB:Extended \
             --era Run3_2025 \
             --filein file:$ALL_FILES \
             --fileout file:step2.root \
             --nThreads 24 \
             --python_filename dqm.py \
             --no_exec

cat <<@EOF >> dqm.py 
# import beamspot
from RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi import onlineBeamSpotProducer as _onlineBeamSpotProducer
process.hltOnlineBeamSpot = _onlineBeamSpotProducer.clone()
process.scoutingCollectionMonitor.onlineMetaDataDigis = "hltOnlineMetaDataDigis"
process.scoutingCollectionMonitor.onlyScouting = True
process.dqmoffline_step = cms.EndPath(process.hltOnlineBeamSpot+process.hltScoutingCollectionMonitor+process.hltScoutingTrackMonitor)
@EOF

cmsRun dqm.py >& dqm.log

cmsDriver.py step3 -s HARVESTING:@standardDQM \
         --conditions 160X_dataRun3_HLT_v1 \
         --data \
         --geometry DB:Extended \
         --scenario pp \
         --filetype DQM \
         --era Run3_2025 \
         -n 10000 \
         --filein file:step2.root \
         --fileout file:step3.root \
         --no_exec

cmsRun step3_HARVESTING.py >& harvesting.log
