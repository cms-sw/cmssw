import FWCore.ParameterSet.Config as cms
process = cms.Process('ANALYSIS')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#process.MessageLogger.messages = cms.untracked.PSet(
#        threshold =  cms.untracked.vstring('DEBUG')
#)

process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("MagneticField.Engine.autoMagneticFieldProducer_cfi")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['startup']

# Specify IdealMagneticField ESSource (needed for CMSSW 730)

#load the response corrections calculator
process.load('Calibration.HcalCalibAlgos.gammaJetAnalysis_cfi')
#  needed for nonCHS
process.load('JetMETCorrections.Configuration.JetCorrectionProducers_cff')

# run over files
process.GammaJetAnalysis.rootHistFilename = cms.string('PhoJet_tree_CHS_ported.root')
process.GammaJetAnalysis.doPFJets = cms.bool(True)
process.GammaJetAnalysis.doGenJets = cms.bool(True)

# trigger names should not end with '_'
process.GammaJetAnalysis.photonTriggers = cms.vstring(
    'HLT_Photon20_CaloIdVL_IsoL','HLT_Photon30_CaloIdVL_IsoL',
    'HLT_Photon50_CaloIdVL_IsoL','HLT_Photon75_CaloIdVL_IsoL',
    'HLT_Photon90_CaloIdVL_IsoL','HLT_Photon135',
    'HLT_Photon150','HLT_Photon160')
# triggers for CMSSW 730
process.GammaJetAnalysis.photonTriggers += cms.vstring(
    'HLT_Photon22', 'HLT_Photon30', 'HLT_Photon36',
    'HLT_Photon50', 'HLT_Photon75',
    'HLT_Photon90', 'HLT_Photon120', 'HLT_Photon175',
    'HLT_Photon250_NoHE', 'HLT_Photon300_NoHE'
)
# to disable photonTriggers assign an empty vstring
#process.GammaJetAnalysis.photonTriggers = cms.vstring()

# a clone without CHS
process.GammaJetAnalysis_noCHS= process.GammaJetAnalysis.clone()
process.GammaJetAnalysis_noCHS.rootHistFilename = cms.string('PhoJet_tree_nonCHS_ported.root')
# for 7XY use ak4* instead of ak5
process.GammaJetAnalysis_noCHS.pfJetCollName = cms.string('ak4PFJets')
process.GammaJetAnalysis_noCHS.pfJetCorrName = cms.string('ak4PFL2L3')

process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring(
#    '/store/relval/CMSSW_7_3_0/RelValPhotonJets_Pt_10_13/GEN-SIM-RECO/MCRUN2_73_V7-v1/00000/522CE329-7B81-E411-B6C3-0025905A6110.root',
#    '/store/relval/CMSSW_7_3_0/RelValPhotonJets_Pt_10_13/GEN-SIM-RECO/MCRUN2_73_V7-v1/00000/5279D224-7B81-E411-BCAA-002618943930.root'
      'file:../../HcalAlCaRecoProducers/test/gjet.root'
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
#process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG') # does not work?
process.MessageLogger.cerr.FwkReport.reportEvery=cms.untracked.int32(500)
process.MessageLogger.cerr.FwkReport.reportEvery=cms.untracked.int32(1)
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )

# Load pfNoPileUP
#process.load("CommonTools.ParticleFlow.pfNoPileUp_cff")
#process.load("CommonTools.ParticleFlow.PF2PAT_cff")
#from RecoJets.JetProducers.ak5PFJets_cfi import *
#process.ak5PFJetsCHS = ak5PFJets.clone(
#    src = cms.InputTag("pfNoPileUp")
#)

#process.load('Calibration.HcalCalibAlgos.gammaJetAnalysis_CHSJECs_cff')

# specify 'workOnAOD=2' to apply tokens from GammaJetProd producer
process.GammaJetAnalysis.workOnAOD = cms.int32(2)
process.GammaJetAnalysis.doGenJets = cms.bool(False)
process.GammaJetAnalysis.debug = cms.untracked.int32(1)

process.p = cms.Path(
#    process.PF2PAT
#    +process.ak5PFJetsCHS
    process.GammaJetAnalysis
)
