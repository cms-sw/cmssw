# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

readFiles = cms.untracked.vstring()
readFiles.extend( [
	'/store/data/BeamCommissioning08/BeamHalo/RECO/CRUZET4_V4P_CSCSkim_trial_v3/0000/00BEE8CD-1181-DD11-8F58-001A4BA82F4C.root'
] );
  
process.source = cms.Source("PoolSource",
	fileNames = readFiles
)

process.TFileService = cms.Service("TFileService",
	fileName = cms.string("output.root"),
	closeFileFast = cms.untracked.bool(True)
)

process.analyze = cms.EDAnalyzer("CSCSharesInputTest",	
	CSCRecHitCollection = cms.InputTag("csc2DRecHits","","Rec"),
	MuonCollection = cms.InputTag("STAMuons","","Rec")
)



process.p = cms.Path(process.analyze)
