import FWCore.ParameterSet.Config as cms

process = cms.Process("ANTEST")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'IDEAL_31X::All'
process.GlobalTag.globaltag = 'MC_31X_V5::All'

process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")

process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_cff")
process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrkNoHLT_cff")
process.IsoProd.MaxTrackEta = cms.double(3.0)

process.isoHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(200)

process.load("Calibration.HcalCalibAlgos.isoAnalyzer_cfi")
#process.isoAnalyzer.AxB = cms.string("7x7")
process.isoAnalyzer.AxB = cms.string("Cone")
process.isoAnalyzer.calibrationConeSize = cms.double(40.)

process.load("HLTrigger.Timer.timer_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)
process.source = cms.Source("PoolSource",
    fileNames =
cms.untracked.vstring(

'rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_312/pi50_1.root',
#'rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_310pre10/pi50_1.root',
#'file:/afs/cern.ch/user/s/sergeant/scratch0/2008/myRawToReco_IsoTr_FullFED.root'
#        'rfio:/castor/cern.ch/user/s/safronov/forIsoTracksFromReco.root'
#'file:/afs/cern.ch/user/s/sergeant/scratch0/2009/CMSSW_3_1_0/src/Configuration/GenProduction/test/ALCARECOHcalCalIsoTrk.root'

)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.TimerService = cms.Service("TimerService",
    useCPUtime = cms.untracked.bool(True)
)

process.TimerService = cms.Service("TimerService", useCPUtime = cms.untracked.bool(True))
process.pts = cms.EDFilter("PathTimerInserter")
process.PathTimerService = cms.Service("PathTimerService")

#Use this Path to run the code on RECO data sets (such as single pions produced by Salavat):
process.AnalIsoTrTest = cms.Path(process.seqALCARECOHcalCalIsoTrkNoHLT*process.isoAnalyzer)

#Use this Path instead to run it on ALCARECO format data:
#process.AnalIsoTrTest = cms.Path(process.isoAnalyzer)


