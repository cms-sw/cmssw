#!/bin/bash -ex

LOCALPATH='/eos/cms/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_STD_D121_RegeneratedGS_PU-v1/2590000/'
echo "Input source: |${LOCALPATH}|"
LOCALFILES=$(ls -1 ${LOCALPATH})
ALL_FILES=""
for f in ${LOCALFILES[@]}; do
    ALL_FILES+="file:${LOCALPATH}/${f},"
done

# Remove the last character
ALL_FILES="${ALL_FILES%?}"
echo "Discovered files: $ALL_FILES"

cmsDriver.py step2 -s L1P2GT,HLT:75e33_timing,NANO:@Phase2HLT \
             --conditions auto:phase2_realistic_T35 \
             --datatier NANOAODSIM \
             -n 100 \
             --eventcontent NANOAODSIM \
             --geometry ExtendedRun4D110 \
             --era Phase2C17I13M9 \
             --filein $ALL_FILES \
             --nThreads 1 \
             --process HLTX \
             --no_exec \
             --inputCommands='keep *, drop *_hlt*_*_HLT, drop trigger*_*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT'

cat <<@EOF >> step2_L1P2GT_HLT_NANO.py
process.NanoHltTables = cms.Sequence(process.hltTriggerObjP4Table)
@EOF

cmsRun step2_L1P2GT_HLT_NANO.py > step2.log  2>&1
