from PhysicsTools.PatAlgos.patTemplate_cfg import *

## load tau sequences up to selectedPatJets
process.load("PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff")
patAlgosToolsTask.add(process.makePatJetsTask)

process.load("PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi")
patAlgosToolsTask.add(process.selectedPatJets)

process.load('RecoBTag/Configuration/RecoBTag_cff')
patAlgosToolsTask.add(process.btaggingTask)

## Events to process
process.maxEvents.input = 10

## Input files
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      '/store/relval/CMSSW_7_6_0_pre3/RelValTTbar_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v2-v1/00000/2E84FB77-FC41-E511-9B44-0025905A612C.root',
      '/store/relval/CMSSW_7_6_0_pre3/RelValTTbar_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v2-v1/00000/3A7D247C-FC41-E511-BEE0-002618943981.root',
    )
)

## Output file
process.out.fileName = 'validate_ctag_pat.root'

