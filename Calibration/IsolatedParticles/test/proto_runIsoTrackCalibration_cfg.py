import FWCore.ParameterSet.Config as cms

process = cms.Process("IsoTrackCalibration")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
        'file:/uscms/home/guptar/3DayLifetime/file.root'
#'        root://xrootd.unl.edu//store/mc/Spring14dr/QCD_Pt-15to3000_Tune4C_Flat_13TeV_pythia8/GEN-SIM-RECO/PU20bx25_POSTLS170_V5-v1/00000/0009CDC8-CCD6-E311-A9AD-90E6BAE8CC37.root'
    )
)

process.load('Calibration.IsolatedParticles.isoTrackCalibration_cfi')
process.IsoTrackCalibration.Verbosity = 99
process.IsoTrackCalibration.Triggers  = ['HLT_PFJet80_v10']
process.IsoTrackCalibration.IsItQCDMC = True
process.IsoTrackCalibration.IsItAOD   = False

process.TFileService = cms.Service("TFileService",
   fileName = cms.string('output.root')
)
process.p = cms.Path(process.IsoTrackCalibration)
