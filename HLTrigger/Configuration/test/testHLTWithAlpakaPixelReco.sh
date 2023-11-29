#!/bin/bash

https_proxy=http://cmsproxy.cms:3128/ \
hltGetConfiguration /frozen/2023/2e34/v1.2/HLT \
  --globaltag 130X_dataRun3_HLT_v2 \
  --data \
  --unprescale \
  --output all \
  --max-events 200 \
  --paths DQM_PixelReco*,*DQMGPUvsCPU* \
  --input /store/data/Run2023C/EphemeralHLTPhysics0/RAW/v1/000/368/822/00000/6e1268da-f96a-49f6-a5f0-89933142dd89.root \
  --customise \
HLTrigger/Configuration/customizeHLTforPatatrack.customizeHLTforPatatrack,\
HLTrigger/Configuration/customizeHLTforPatatrack.customizeHLTforDQMGPUvsCPUPixel \
  > hlt.py

cat <<EOF >> hlt.py
process.options.numberOfThreads = 1
process.options.numberOfStreams = 0

del process.MessageLogger
process.load('FWCore.MessageLogger.MessageLogger_cfi')

# assign only DQM_PixelReconstruction_v to the DQMGPUvsCPU Primary Dataset
process.hltDatasetDQMGPUvsCPU.triggerConditions = ['DQM_PixelReconstruction_v*']

# remove FinalPaths running OutputModules, except for the DQMGPUvsCPU and DQM ones
finalPathsToRemove = []
for fpath in process.finalpaths_():
    if fpath not in ['DQMOutput', 'DQMGPUvsCPUOutput']:
        finalPathsToRemove += [fpath]
for fpath in finalPathsToRemove:
    process.__delattr__(fpath)

# do not produce output of DQM stream (not needed)
del process.hltOutputDQM

# rename DQMIO output file
process.dqmOutput.fileName = '___JOBNAME____DQMIO.root'

# rename output of DQMGPUvsCPU stream
process.hltOutputDQMGPUvsCPU.fileName = '___JOBNAME___.root'
EOF

JOBNAME=hlt0
sed "s|___JOBNAME___|${JOBNAME}|" hlt.py > "${JOBNAME}".py
edmConfigDump --prune "${JOBNAME}".py > "${JOBNAME}"_dump.py
echo "${JOBNAME}" ... && cmsRun "${JOBNAME}".py &> "${JOBNAME}".log
