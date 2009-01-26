#!/bin/csh -f

set cfgFile = $1
set runNum = $2
echo CFG FILE: ${cfgFile}
setenv Outdir /castor/cern.ch/cms/store/cmscaf/L1Trigger/L1Prompt
setenv runDir ${CMSSW_BASE}/src/L1TriggerOffline/L1Analyzer/test
setenv tmpDir `pwd`
echo TEMPORARY DIRECTORY: ${tmpDir}
echo RUN DIRECTORY: ${runDir}




cd ${runDir}

eval `scramv1 runtime -csh`

cmsRun ${cfgFile} 
rfcp ${runNum}.root ${Outdir}
rfcp ${cfgFile} ${Outdir}
rfcp ${runNum}.log ${Outdir}
rm -f ${runNum}.root
rm -f ${runNum}.log
rm -f ${cfgFile}
rm -f core.*
