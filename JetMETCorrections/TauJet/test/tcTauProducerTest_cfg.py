import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("test")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#	'file:/tmp/slehti/A161_RECO.root'
#	'/store/relval/CMSSW_3_1_0_pre11/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/A2CC8319-F064-DE11-B0C6-00304876A0FF.root'
	'rfio:/castor/cern.ch/user/s/slehti/testData/Ztautau_GEN_SIM_RECO_MC_31X_V2_preproduction_311_v1.root'
    )
)

process.load("FWCore/MessageService/MessageLogger_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'IDEAL_V1::All'
process.GlobalTag.globaltag = 'STARTUP31X_V1::All'

process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CMSCommonData.cmsAllGeometryXML_cfi")
process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

process.load("JetMETCorrections/TauJet/TCRecoTauProducer_cfi")

process.runTCTauProducer = cms.Path(
	process.tcRecoTauProducer
)

process.TESTOUT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        "keep *"
    ),
    fileName = cms.untracked.string('file:testout.root')
)
process.outpath = cms.EndPath(process.TESTOUT)
