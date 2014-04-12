import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(1),
	
    fileNames = cms.untracked.vstring(

        '/store/relval/CMSSW_3_2_1/RelValSingleMuPt100/GEN-SIM-RECO/MC_31X_V3-v1/0006/DC15F12B-9477-DE11-B1E0-000423D98C20.root',
        '/store/relval/CMSSW_3_2_1/RelValSingleMuPt100/GEN-SIM-RECO/MC_31X_V3-v1/0006/40D6FEFD-8F77-DE11-95A7-001D09F27067.root',
        '/store/relval/CMSSW_3_2_1/RelValSingleMuPt100/GEN-SIM-RECO/MC_31X_V3-v1/0005/50EE1208-8177-DE11-8B17-001D09F231B0.root'

    )
)

process.maxEvents = cms.untracked.PSet(
	input=cms.untracked.int32(100)
)

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
from Configuration.StandardSequences.Reconstruction_cff import *

process.muonAnalyzer = cms.EDFilter("MuonTimingValidator",
  TKtracks = cms.untracked.InputTag("generalTracks"),
  STAtracks = cms.untracked.InputTag("standAloneMuons"),
  Muons = cms.untracked.InputTag("muons"),
  nbins = cms.int32(60),
  PtresMax = cms.double(2000.0),
  Timing = cms.untracked.InputTag("muontiming"),
  simPtMin = cms.double(5.0),
  PtresMin = cms.double(-1000.0),
  PtCut = cms.double(1.0),
  etaMax = cms.double(2.4),
  etaMin = cms.double(0.0),
  PlotScale = cms.double(1.0),
  DTcut  = cms.int32(8),
  CSCcut = cms.int32(4),
  open = cms.string('recreate'),
  out = cms.string('test_timing.root')
)

process.prefer("GlobalTag")

process.GlobalTag.globaltag = 'MC_3XY_V14::All'
#process.GlobalTag.globaltag = 'STARTUP_V7::All'

process.p = cms.Path(muontiming)

process.mutest = cms.Path(process.muonAnalyzer)

process.schedule = cms.Schedule(process.p,process.mutest)	
# process.schedule = cms.Schedule(process.mutest)	

