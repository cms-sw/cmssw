import FWCore.ParameterSet.Config as cms
process = cms.Process('ANALYSIS')

process.load('Configuration.StandardSequences.Services_cff')
# Specify IdealMagneticField ESSource (needed for CMSSW 730)
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run1_data']

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.GammaJetAnalysis=dict()
process.MessageLogger.cerr.FwkReport.reportEvery=cms.untracked.int32(1000)

#load the analyzer
process.load('Calibration.HcalCalibAlgos.gammaJetAnalysis_cfi')
# load energy corrector
process.load('JetMETCorrections.Configuration.JetCorrectionProducers_cff')

# run over files
process.GammaJetAnalysis.rootHistFilename = cms.string('PhoJet_tree_CHS_data2012_noGJetProd.root')
process.GammaJetAnalysis.doPFJets = cms.bool(True)
process.GammaJetAnalysis.doGenJets = cms.bool(False)

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

process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring(
        'file:/tmp/andriusj/Run2012A_Photon_22Jan2013-002618943913.root'
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )

# Load jets and pfNoPileUP
process.load('RecoJets.Configuration.RecoPFJets_cff')
process.load('CommonTools.ParticleFlow.PF2PAT_cff')
process.load("CommonTools.ParticleFlow.pfNoPileUp_cff")

process.seq_ak4PFCHS= cms.Sequence( process.particleFlowPtrs *
                                    process.pfNoPileUpJMESequence *
                                    process.ak4PFJetsCHS )

# adapt input collections
process.GammaJetAnalysis.photonCollName= cms.string("photons")
process.GammaJetAnalysis.electronCollName= cms.string("gsfElectrons")
process.GammaJetAnalysis.photonIdLooseName= cms.InputTag("PhotonIDProd","PhotonCutBasedIDLoose")
process.GammaJetAnalysis.photonIdTightName= cms.InputTag("PhotonIDProd","PhotonCutBasedIDTight")

# name of the process that used the GammaJetProd producer
#process.GammaJetAnalysis.prodProcess = cms.untracked.string('MYGAMMAJET')
# specify 'workOnAOD=2' to apply tokens from GammaJetProd producer
process.GammaJetAnalysis.workOnAOD = cms.int32(0)
process.GammaJetAnalysis.doGenJets = cms.bool(False)
process.GammaJetAnalysis.debug     = cms.untracked.int32(0)

process.p = cms.Path(
    process.seq_ak4PFCHS *
    process.GammaJetAnalysis
)
