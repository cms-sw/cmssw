condorJobTemplate="""#!/bin/tcsh

set curDir=$PWD
echo $curDir
cd {base}/../..
eval `scramv1 runtime -csh`

cd $curDir

xrdcp {inputFile} reco.root

cmsRun {base}/test/cfgTemplate/apeEstimator_cfg.py {inputCommands}

rm reco.root
"""

condorSubTemplate="""
Executable = {jobFile}
Universe = vanilla
Output = {outputFile}
Error  = {errorFile}
Log  = {logFile}
request_memory = 2000M
request_disk = 400M
batch_name = {jobName}
+JobFlavour = "workday"
Queue Arguments from (
{arguments})
"""


condorSubTemplateCAF="""
Executable = {jobFile}
Universe = vanilla
Output = {outputFile}
Error  = {errorFile}
Log  = {logFile}
request_memory = 2000M
request_disk = 400M
batch_name = {jobName}
+JobFlavour = "workday"
+AccountingGroup = "group_u_CMS.CAF.ALCA" 
Queue Arguments from (
{arguments})
"""

condorArgumentTemplate="""{fileNumber} {inputFile}
"""

submitCondorTemplate="""
condor_submit {subFile}
"""

killJobTemplate="condor_rm {jobId}"

summaryTemplate="cmsRun $CMSSW_BASE/src/Alignment/APEEstimation/test/cfgTemplate/apeEstimatorSummary_cfg.py {inputCommands}"

mergeTemplate="hadd {path}/allData.root {inputFiles}"

localSettingTemplate="cmsRun $CMSSW_BASE/src/Alignment/APEEstimation/test/cfgTemplate/apeLocalSetting_cfg.py {inputCommands}"

conditionsFileHeader="""
import FWCore.ParameterSet.Config as cms
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import poolDBESSource
def applyConditions(process):
"""

conditionsTemplate="""
    process.my{record}Conditions = poolDBESSource.clone(
    connect = cms.string('{connect}'),
    toGet = cms.VPSet(cms.PSet(record = cms.string('{record}'),
                            tag = cms.string('{tag}')
                            )
                    )
    )
    process.prefer_my{record}Conditions = cms.ESPrefer("PoolDBESSource", "my{record}Conditions")
"""
