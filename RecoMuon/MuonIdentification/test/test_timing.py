import FWCore.ParameterSet.Config as cms

process = cms.Process("test")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")

#process.prefer("GlobalTag")
#process.GlobalTag.globaltag = 'IDEAL_V11::All'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20) )

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_3_1_0_pre4/RelValSingleMuPt100/GEN-SIM-RECO/IDEAL_30X_v1/0003/1C780F8D-AB16-DE11-B436-001617E30E2C.root'
#'file:/data/ptraczyk/relval/310_100/80B3394A-180A-DE11-BC92-000423D99AAE.root'
  
  )
)

process.load("RecoMuon.MuonIdentification.muonTiming_cfi")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myOutputFile.root')
)

#process.p = cms.Path(process.siPixelRecHits+process.siStripMatchedRecHits+process.ckftracks+process.muonrecowith_TeVRefinemen+process.muontiming)
process.p = cms.Path(process.muontiming)

process.e = cms.EndPath(process.out)
