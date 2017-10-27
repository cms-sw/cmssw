import FWCore.ParameterSet.Config as cms

process = cms.Process("RaddamMuon")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")  
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoJets.Configuration.CaloTowersES_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag='92X_dataRun2_Prompt_v5'

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("Calibration.HcalCalibAlgos.hcalHBHEMuon_cfi")

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('HBHEMuon')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'root://cms-xrd-global.cern.ch//store/data/Run2017B/SingleMuon/RECO/PromptReco-v2/000/298/853/00000/02946E05-6868-E711-ADDE-02163E011BFF.root',
        'root://cms-xrd-global.cern.ch//store/data/Run2017B/SingleMuon/RECO/PromptReco-v2/000/298/678/00000/C0C0C0B0-A466-E711-AE46-02163E019E8C.root',
#       'file:/afs/cern.ch/work/a/amkalsi/public/ForSunandaDa/C0C0C0B0-A466-E711-AE46-02163E019E8C.root',
#       'root://xrootd.unl.edu//store/data/Run2017B/SingleMuon/RECO/PromptReco-v2/000/298/678/00000/C0C0C0B0-A466-E711-AE46-02163E019E8C.root'
#        'root://xrootd.unl.edu//store/mc/Phys14DR/DYToMuMu_M-50_Tune4C_13TeV-pythia8/GEN-SIM-RECO/PU20bx25_tsg_castor_PHYS14_25_V1-v1/10000/184C1AC9-A775-E411-9196-002590200824.root'
        )
                            )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("Validation.root")
)

process.hcalTopologyIdeal.MergePosition = False
process.hcalHBHEMuon.UseRaw = False
process.hcalHBHEMuon.UnCorrect = True
process.hcalHBHEMuon.GetCharge = True
process.hcalHBHEMuon.CollapseDepth = False
process.hcalHBHEMuon.IsItPlan1 = True
process.hcalHBHEMuon.IgnoreHECorr = False
process.hcalHBHEMuon.IsItPreRecHit = True
process.hcalHBHEMuon.MaxDepth = 7
process.hcalHBHEMuon.LabelHBHERecHit = cms.InputTag("hbheprereco")
process.hcalHBHEMuon.Verbosity = 0

process.p = cms.Path(process.hcalHBHEMuon)
