import FWCore.ParameterSet.Config as cms
process = cms.Process('ANALYSIS')

process.load('Configuration.StandardSequences.Services_cff')
# Specify IdealMagneticField ESSource (needed for CMSSW 730)
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.categories+=cms.untracked.vstring('GammaJetAnalysis')
#process.MessageLogger.cerr.FwkReport.reportEvery=cms.untracked.int32(1000)

#load the response corrections calculator
process.load('Calibration.HcalCalibAlgos.gammaJetAnalysis_cfi')
#  needed for nonCHS
#process.load('JetMETCorrections.Configuration.JetCorrectors_cff')
process.load('JetMETCorrections.Configuration.JetCorrectionProducers_cff')

# run over files
process.GammaJetAnalysis.rootHistFilename = cms.string('PhoJet_tree_CHS_noGJetProd.root')
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
process.GammaJetAnalysis_noCHS.rootHistFilename = cms.string('PhoJet_tree_nonCHS_noGJetProd.root')
# for 7XY use ak4* instead of ak5
process.GammaJetAnalysis_noCHS.pfJetCollName = cms.string('ak4PFJets')
process.GammaJetAnalysis_noCHS.pfJetCorrName = cms.string('ak4PFL2L3')

process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring(
#   'file:/tmp/andriusj/6EC8FCC8-E2A8-E411-9506-002590596468.root'
        '/store/relval/CMSSW_7_4_0_pre6/RelValPhotonJets_Pt_10_13/GEN-SIM-RECOMCRUN2_74_V1-v1/00000/6EC8FCC8-E2A8-E411-9506-002590596468.root'
    )
)

#To have the same number of histograms, do not run over GenJets
#process.GammaJetAnalysis.doGenJets = cms.bool(False)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.FwkReport.reportEvery=cms.untracked.int32(1000)
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )

process.p = cms.Path(
    process.GammaJetAnalysis
)
