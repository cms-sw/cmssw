import FWCore.ParameterSet.Config as cms

process = cms.Process("DIJETSANALYSIS")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']

process.load("RecoJets.Configuration.CaloTowersRec_cff")                                                           
process.towerMaker.ecalInputs = cms.VInputTag(cms.InputTag("DiJProd","DiJetsEcalRecHitCollection"))
process.towerMaker.hbheInput = cms.InputTag("HitsReCalibration","DiJetsHBHEReRecHitCollection")
process.towerMaker.hoInput = cms.InputTag("HitsReCalibration","DiJetsHOReRecHitCollection")
process.towerMaker.hfInput = cms.InputTag("HitsReCalibration","DiJetsHFReRecHitCollection")
process.towerMakerWithHO.ecalInputs = cms.VInputTag(cms.InputTag("DiJProd", "DiJetsEcalRecHitCollection"))
process.towerMakerWithHO.hbheInput = cms.InputTag("HitsReCalibration","DiJetsHBHEReRecHitCollection")
process.towerMakerWithHO.hoInput = cms.InputTag("HitsReCalibration","DiJetsHOReRecHitCollection")
process.towerMakerWithHO.hfInput = cms.InputTag("HitsReCalibration","DiJetsHFReRecHitCollection")

# The file imported by this line was deleted from CMSSW in 2017.
# This is the first thing that fails if you try to run this.
# This is not even parseable by Python.
process.load("RecoJets.JetProducers.ic5CaloJets_cfi")

process.iterativeCone5CaloJets.doPVCorrection = cms.bool(False)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = 
cms.untracked.vstring(
'/store/relval/CMSSW_3_3_0_pre5/RelValQCD_Pt_80_120/ALCARECO/STARTUP31X_V7_StreamHcalCalDijets-v1/0004/B421101A-6AAB-DE11-A988-001D09F2A465.root'
)
)

process.es_ascii2 = cms.ESSource("HcalTextCalibrations",
                                 input = cms.VPSet(cms.PSet(
            object = cms.string('RespCorrs'),
            file = cms.FileInPath('Calibration/HcalCalibAlgos/data/calibConst_IsoTrk_testCone_26.3cm.txt')
    )),
    appendToDataLabel = cms.string('recalibrate')
)

process.prefer("es_ascii2")
process.HitsReCalibration = cms.EDProducer("HitReCalibrator",
    hbheInput = cms.InputTag("DiJProd","DiJetsHBHERecHitCollection"),
    hfInput = cms.InputTag("DiJProd","DiJetsHFRecHitCollection"),
    hoInput = cms.InputTag("DiJProd","DiJetsHORecHitCollection")
)

# I did not add the parameter "JetCorrections" when I upgraded to use the
# new reco::JetCorrector because the pre-existing version already did not
# have the "pfJetCorrName" parameter to identify which corrections to use
# for the deprecated JetCorrector class.
# This means the configuration using the old JetCorrectors was already broken.
# There was no information available to identify which corrections are desired.
# So at least that will need to be fixed before this
# configuration will run. This configuration also looks like it
# has not been run since ~2010 and there are many other broken things that
# will also need fixing before it will run. For example, there are several other
# configuration parameters missing just for this module.
process.DiJetAnalysis = cms.EDAnalyzer("DiJetAnalyzer",
    hbheInput = cms.InputTag("HitsReCalibration","DiJetsHBHEReRecHitCollection"),
    HistOutFile = cms.untracked.string('hi.root'),
    hfInput = cms.InputTag("HitsReCalibration","DiJetsHFReRecHitCollection"),
    hoInput = cms.InputTag("HitsReCalibration","DiJetsHOReRecHitCollection"),
    ecInput = cms.InputTag("DiJProd","DiJetsEcalRecHitCollection"),
    jetsInput = cms.InputTag("iterativeCone5CaloJets")
)

process.TFileService = cms.Service("TFileService",
   fileName = cms.string('hi.root')
)

#process.DiJetsRecoPool = cms.OutputModule("PoolOutputModule",
#    outputCommands = cms.untracked.vstring('drop *', 
#        'keep *_DiJetsReco_*_*'),
#    fileName = cms.untracked.string('/tmp/andrey/tmp.root')
#)

process.p = cms.Path(process.HitsReCalibration*process.caloTowersRec*process.iterativeCone5CaloJets*process.DiJetAnalysis)

