## THIS IS CURRENTLY COMPATIBLE ONLY WITH SINGLE TOP MODULES 
## since the b-tagging algorithms are here re-run with PFJets as input

import FWCore.ParameterSet.Config as cms

process = cms.Process('B2GDQM')

## imports of standard configurations
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

## input file(s) for testing
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_7_2_0_pre5/RelValRSKKGluon_m3000GeV_13/GEN-SIM-RECO/POSTLS172_V3-v1/00000/28A38647-FA30-E411-8A85-0025905A608C.root',
'/store/relval/CMSSW_7_2_0_pre5/RelValRSKKGluon_m3000GeV_13/GEN-SIM-RECO/POSTLS172_V3-v1/00000/943D26E9-7B30-E411-8D18-0025905A48D8.root',
'/store/relval/CMSSW_7_2_0_pre5/RelValRSKKGluon_m3000GeV_13/GEN-SIM-RECO/POSTLS172_V3-v1/00000/CE57EA42-7830-E411-8A2E-0025905A6132.root'
     )
)

## number of events
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.load("DQM.Physics.B2GDQM_cfi")



## output
process.output = cms.OutputModule("PoolOutputModule",
  fileName       = cms.untracked.string('b2gDQM.root'),
  outputCommands = cms.untracked.vstring(
    'drop *_*_*_*',
    'keep *_*_*_B2GDQM'
    ),
  splitLevel     = cms.untracked.int32(0),
  dataset = cms.untracked.PSet(
    dataTier   = cms.untracked.string(''),
    filterName = cms.untracked.string('')
  )
)

## check the event content
process.content = cms.EDAnalyzer("EventContentAnalyzer")

## path definitions
process.p      = cms.Path(
    process.B2GDQM
    
)
process.endjob = cms.Path(
    process.endOfProcess
)

process.fanout = cms.EndPath(
    process.output
)


## schedule definition
process.schedule = cms.Schedule(
    process.p,
    process.endjob,
    process.fanout
)
