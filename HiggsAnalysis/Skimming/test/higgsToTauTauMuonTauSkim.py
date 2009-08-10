# Skim for MSSM Higgs->tautau->muontau Events
# Created by Monica Vazquez Acosta, 20/05/2009

import FWCore.ParameterSet.Config as cms
process = cms.Process("HTauTauMuonTauSkim")

process.maxEvents = cms.untracked.PSet(
         input = cms.untracked.int32(-1)
)


process.load("HiggsAnalysis.Skimming.higgsToTauTau_MuonTau_SkimPaths_cff")
process.load("HiggsAnalysis.Skimming.higgsToTauTau_MuonTau_OutputModule_cff")

process.load("FWCore/MessageService/MessageLogger_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.source = cms.Source("PoolSource",
    skipBadFiles = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
      'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_1_2/RelValZTT/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/7AAAAFA8-CA78-DE11-8FE2-001D09F241B4.root',
      'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_1_2/RelValZTT/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/7C4B7106-B378-DE11-9C6E-000423D94990.root',
      'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_1_2/RelValZTT/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/9408B54D-CB78-DE11-9AEB-001D09F2503C.root',
      'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_1_2/RelValZTT/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/A4DD1FAE-B178-DE11-B608-001D09F24EAC.root'
    )
)


configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/HiggsAnalysis/Skimming/test/higgsToTauTauMuonTauSkim.py,v $'),
    annotation = cms.untracked.string('Skim for heavy MSSM Higgs> tau tau > muon tau events')
)

process.outpath = cms.EndPath(process.higgsToTauTauMuonTauOutputModuleRECOSIM)


