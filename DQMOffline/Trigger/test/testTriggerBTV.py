import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMOffline.Trigger.BTVHLTOfflineSource_cfi")
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.globaltag = '81X_mcRun2_asymptotic_v5'
process.prefer("GlobalTag")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'root://cms-xrd-global.cern.ch//store/relval/CMSSW_8_1_0_pre10/RelValTTbarLepton_13/GEN-SIM-RECO/81X_mcRun2_asymptotic_v5_hipoffHLT-v1/00000/2AB0436C-AB6E-E611-9663-0CC47A4C8ECA.root',
        'root://cms-xrd-global.cern.ch//store/relval/CMSSW_8_1_0_pre10/RelValTTbarLepton_13/GEN-SIM-RECO/81X_mcRun2_asymptotic_v5_hipoffHLT-v1/00000/9695BFEF-AA6E-E611-8A07-0CC47A4D76B2.root' 
    ),
    secondaryFileNames = cms.untracked.vstring(
        'root://cms-xrd-global.cern.ch//store/relval/CMSSW_8_1_0_pre10/RelValTTbarLepton_13/GEN-SIM-DIGI-RAW-HLTDEBUG/81X_mcRun2_asymptotic_v5_hipoffHLT-v1/00000/20E851A4-A56E-E611-83E3-0CC47A78A4BA.root',
        'root://cms-xrd-global.cern.ch//store/relval/CMSSW_8_1_0_pre10/RelValTTbarLepton_13/GEN-SIM-DIGI-RAW-HLTDEBUG/81X_mcRun2_asymptotic_v5_hipoffHLT-v1/00000/366730AF-A46E-E611-90C3-0CC47A78A456.root',
        'root://cms-xrd-global.cern.ch//store/relval/CMSSW_8_1_0_pre10/RelValTTbarLepton_13/GEN-SIM-DIGI-RAW-HLTDEBUG/81X_mcRun2_asymptotic_v5_hipoffHLT-v1/00000/D63E50B3-A46E-E611-BCD8-0CC47A4D75F8.root',
        'root://cms-xrd-global.cern.ch//store/relval/CMSSW_8_1_0_pre10/RelValTTbarLepton_13/GEN-SIM-DIGI-RAW-HLTDEBUG/81X_mcRun2_asymptotic_v5_hipoffHLT-v1/00000/E2C277AB-A46E-E611-BF0C-0025905B8562.root',
        'root://cms-xrd-global.cern.ch//store/relval/CMSSW_8_1_0_pre10/RelValTTbarLepton_13/GEN-SIM-DIGI-RAW-HLTDEBUG/81X_mcRun2_asymptotic_v5_hipoffHLT-v1/00000/F26AFCAF-A46E-E611-906B-0CC47A745298.root',
        'root://cms-xrd-global.cern.ch//store/relval/CMSSW_8_1_0_pre10/RelValTTbarLepton_13/GEN-SIM-DIGI-RAW-HLTDEBUG/81X_mcRun2_asymptotic_v5_hipoffHLT-v1/00000/F2899FA2-A46E-E611-B2A6-0CC47A4D769A.root'
    ),
)

print "File defined"

process.p = cms.EndPath(
process.BTVHLTOfflineSource + process.dqmEnv+process.dqmSaver
)

process.DQMStore.verbose = 0
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'HLT'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

outfile = open('config.py','w')
print >> outfile,process.dumpPython()
outfile.close()
