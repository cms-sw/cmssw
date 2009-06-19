# Skim for heavy H+->taunu background analysis with muon events
# Filter events with HLT single muon + three jets Et > 20, abs(eta) < 2.5
# then produces AODSIM selected events
# Created by S.Lehti
# Tested on 28 May 2009

import FWCore.ParameterSet.Config as cms
process = cms.Process("HChMuonSkim")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(100)
)


process.load("HiggsAnalysis.Skimming.heavyChHiggsToTauNu_SkimPaths_cff")
process.load("HiggsAnalysis.Skimming.heavyChHiggsToTauNu_OutputModule_cff")

process.load("FWCore/MessageService/MessageLogger_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	"file:heavyChHiggsToTauNuSkim.root"
    )
)

configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/HiggsAnalysis/Skimming/test/heavyChHiggsToTauNuMuonBackgroundAnalysisSkim.py,v $'),
    annotation = cms.untracked.string('Skim for heavy H+->tau nu background events with muon selected instead of tau')
)
process.heavyChHiggsToTauNuHLTFilter.HLTPaths = ['HLT_IsoMu15']
process.heavyChHiggsToTauNuFilter.minNumberOfJets = cms.int32(3)

process.outpath = cms.EndPath(process.heavyChHiggsToTauNuOutputModuleRECOSIM)

