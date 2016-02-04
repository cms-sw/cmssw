import FWCore.ParameterSet.Config as cms

process = cms.Process("Validator")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'GR09_R_V5::All'
#process.GlobalTag.globaltag = 'GR09_P_V6::All'
#process.GlobalTag.globaltag = 'MC_31X_V5::All'

process.GlobalTag.globaltag = 'DESIGN_3X_V24::All'
process.prefer("GlobalTag")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(200)

#process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
#from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import hltLevel1GTSeed 
#process.bit40 = hltLevel1GTSeed.clone(
#  L1TechTriggerSeeding = cms.bool(True), 
#  L1SeedsLogicalExpression= cms.string('40 AND NOT (36 OR 37 OR 38 OR 39)' )
#)

process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")
process.load("Configuration.StandardSequences.Generator_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_cff")
process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrkNoHLT_cff")
#process.IsoProd.SkipNeutralIsoCheck = cms.untracked.bool(True)
#process.IsoProd.MinTrackP = cms.double(4.0)

process.isoHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")

process.load("Calibration.HcalCalibAlgos.calib_validator_cfi")
process.ValidationIsoTrk.AxB = cms.string("Cone")
process.ValidationIsoTrk.calibrationConeSize = cms.double(26.2)
process.ValidationIsoTrk.takeAllRecHits = cms.untracked.bool(False)

process.es_ascii2 = cms.ESSource("HcalTextCalibrations",
 appendToDataLabel = cms.string('recalibrate'),
  input = cms.VPSet(
  cms.PSet(object = cms.string('RespCorrs'),
    file = cms.FileInPath('Calibration/HcalCalibAlgos/data/calibConst_IsoTrk_testCone_26.3cm.txt')
          ),
   )
)

process.load("HLTrigger.Timer.timer_cfi")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

#@FNAL: 
#Rel325
#'/store/user/andrey/SinglePions_50GeV_Rel352_v3/SinglePions_50GeV_Rel352_v3/791ecbb28bc75b5af691fc4b56276304/SinglePionMinus_50_1.root',
#'/store/user/andrey/SinglePions_50GeV_Rel352_v3/SinglePions_50GeV_Rel352_v3/791ecbb28bc75b5af691fc4b56276304/SinglePionMinus_50_8.root',     
#'/store/user/andrey/SinglePions_50GeV_Rel352_v3/SinglePions_50GeV_Rel352_v3/791ecbb28bc75b5af691fc4b56276304/SinglePionMinus_50_7.root',
#'/store/user/andrey/SinglePions_50GeV_Rel352_v3/SinglePions_50GeV_Rel352_v3/791ecbb28bc75b5af691fc4b56276304/SinglePionMinus_50_6.root',

#Rel314
'/store/user/andrey/SinglePion_50GeV_314/SinglePion_50GeV_314/0d8aafd1bbf7b6158b7a4e52f0fb00b6/SinglePion_50GeV_314_9.root',

#'/store/data/BeamCommissioning09/MinimumBias/RECO/rereco_FIRSTCOLL_v1/0083/FE5EDBBC-7DD9-DE11-9589-001A92971B64.root',     
#'/store/data/BeamCommissioning09/MinimumBias/RECO/rereco_FIRSTCOLL_v1/0083/FAE7D85F-8AD9-DE11-A342-0026189438C4.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/rereco_FIRSTCOLL_v1/0083/F8EE1B4D-28D9-DE11-BF9C-00261894393B.root',

# Dec Collision ReReco
#'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_336p3_v2/0112/BC80CD7C-94EF-DE11-9134-0024E8767D79.root',        
#'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_336p3_v2/0104/5E9C7099-C2EE-DE11-B85C-001D09675427.root',

#'rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_312/pi50_3.root',

#'rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_310pre10/pi50_1.root',
#'rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_310pre10/pi50_2.root'
#'rfio:/castor/cern.ch/user/a/abdullin/pi300_fullproduction_310pre10/pi300_1.root'
    )
)
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.TimerService = cms.Service("TimerService", useCPUtime = cms.untracked.bool(True))
process.pts = cms.EDProducer("PathTimerInserter")
process.PathTimerService = cms.Service("PathTimerService")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('ValidFile_XX.root')
)

process.p = cms.Path(process.seqALCARECOHcalCalIsoTrkNoHLT*process.ValidationIsoTrk)
#process.p = cms.Path(process.IsoProd*process.ValidationIsoTrk)
#process.p = cms.Path(process.ValidationIsoTrk)



