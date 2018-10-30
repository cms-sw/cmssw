bjobTemplate="""#!/bin/tcsh

cd $CMSSW_BASE/src

eval `scramv1 runtime -csh`

source /afs/cern.ch/cms/caf/setup.csh
cd -

xrdcp {inputFile} reco.root

cmsRun $CMSSW_BASE/src/Alignment/APEEstimation/test/cfgTemplate/apeEstimator_cfg.py {inputCommands}

rm -- "$0"
"""

submitJobTemplate="""
bsub -J {jobName} -e {errorFile} -o {outputFile} -q cmscaf1nd -R "rusage[pool=3000]" tcsh {jobFile}
"""

checkJobTemplate="bjobs -noheader -a -J {jobName}"

killJobTemplate="bkill -J {jobName}"

summaryTemplate="cmsRun $CMSSW_BASE/src/Alignment/APEEstimation/test/cfgTemplate/apeEstimatorSummary_cfg.py {inputCommands}"

mergeTemplate="hadd {path}/allData.root {inputFiles}"

localSettingTemplate="cmsRun $CMSSW_BASE/src/Alignment/APEEstimation/test/cfgTemplate/apeLocalSetting_cfg.py {inputCommands}"

conditionsFileHeader="""
import FWCore.ParameterSet.Config as cms
def applyConditions(process):
"""

conditionsTemplate="""
    import CalibTracker.Configuration.Common.PoolDBESSource_cfi
    process.my{record}Conditions = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
    connect = cms.string('{connect}'),
    toGet = cms.VPSet(cms.PSet(record = cms.string('{record}'),
                            tag = cms.string('{tag}')
                            )
                    )
    )
    process.prefer_my{record}Conditions = cms.ESPrefer("PoolDBESSource", "my{record}Conditions")
"""
