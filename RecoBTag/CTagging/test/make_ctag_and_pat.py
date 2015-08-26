from PhysicsTools.PatAlgos.patTemplate_cfg import *
## switch to uncheduled mode

process.options.allowUnscheduled = cms.untracked.bool(True)
#process.Tracer = cms.Service("Tracer")

## load tau sequences up to selectedPatJets
process.load("PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi")

process.load('RecoBTag/Configuration/RecoBTag_cff')
## process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
## from Configuration.AlCa.GlobalTag import GlobalTag
## process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')

## Events to process
process.maxEvents.input = 1000

## Input files
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      '/store/relval/CMSSW_7_6_0_pre3/RelValTTbar_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v2-v1/00000/2E84FB77-FC41-E511-9B44-0025905A612C.root',
      '/store/relval/CMSSW_7_6_0_pre3/RelValTTbar_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v2-v1/00000/3A7D247C-FC41-E511-BEE0-002618943981.root',
    )
)

## Output file
process.out.fileName = 'validate_ctag_pat.root'


#process.p = cms.Path(
#    process.pfCTagging
#)

