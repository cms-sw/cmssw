# Skim for Light Charged Higgs 
# Created by Nuno Almeida, 15/08/2009

import FWCore.ParameterSet.Config as cms
process = cms.Process("LightChHiggsToTauNuSkim")

process.maxEvents = cms.untracked.PSet(
         input = cms.untracked.int32(100)
)


process.load("HiggsAnalysis.Skimming.lightChHiggsToTauNu_SkimPaths_cff")
process.load("HiggsAnalysis.Skimming.lightChHiggsToTauNu_OutputModule_cff")

process.load("FWCore/MessageService/MessageLogger_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.source = cms.Source("PoolSource",
    skipBadFiles = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
      'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_1_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0006/0CC00E3B-5A78-DE11-A2AB-000423D94A04.root'  #,
     # 'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_1_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0006/307AE787-5678-DE11-AA93-001D09F276CF.root',
     # 'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_1_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0006/3A6D5710-5278-DE11-AB68-000423D98E54.root'

     
    )
)


configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/HiggsAnalysis/Skimming/test/lightChHiggsToTauNu.py,v $'),
    annotation = cms.untracked.string('Skim for light charged Higgs> tau + e/m  events')
)

process.outpath = cms.EndPath(process.lightChHiggsToTauNuOutputModuleRECOSIM)


