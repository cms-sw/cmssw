import FWCore.ParameterSet.Config as cms

process = cms.Process('DIJETANALYSIS')

process.load('FWCore.MessageService.MessageLogger_cfi')

process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run1_mc']

#load the response corrections calculator
process.load('Calibration.HcalCalibAlgos.diJetAnalyzer_cfi')
process.load('JetMETCorrections.Configuration.JetCorrectionProducers_cff')

# run over files

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_80_120_13/GEN-SIM-RECO/MCRUN2_73_V9_71XGENSIM_FIXGT-v1/00000/8E758AAA-4DA2-E411-8068-003048FFCB96.root',
        '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_80_120_13/GEN-SIM-RECO/MCRUN2_73_V9_71XGENSIM_FIXGT-v1/00000/CE0DAE28-56A2-E411-AEFF-003048FFD79C.root',
        '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_80_120_13/GEN-SIM-RECO/MCRUN2_73_V9_71XGENSIM_FIXGT-v1/00000/D4D21D16-56A2-E411-A0C4-0026189438E2.root'
        ))

#print readFiles

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.FwkReport.reportEvery=cms.untracked.int32(1000)
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# Load pfNoPileUP

#process.load("CommonTools.ParticleFlow.pfNoPileUp_cff")
#process.load("CommonTools.ParticleFlow.PF2PAT_cff")
#from RecoJets.JetProducers.ak5PFJets_cfi import *
#process.ak5PFJetsCHS = ak5PFJets.clone(
#    src = cms.InputTag("pfNoPileUp")
#    )
#process.load('HcalClosureTest.Analyzers.calcrespcorr_CHSJECs_cff')

# timing
#process.Timing = cms.Service('Timing')

#process.p = cms.Path(process.pfNoPileUpSequence+process.PF2PAT+process.ak5PFJetsCHS+process.calcrespcorrdijets)
process.p = cms.Path(process.diJetAnalyzer)
