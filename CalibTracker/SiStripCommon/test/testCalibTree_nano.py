
## adapted from produceCalibrationTree_template_cfg.py

import FWCore.ParameterSet.Config as cms
##from CalibTracker.SiStripCommon.shallowTree_test_template import * ## TODO get rid of this one

process = cms.Process('CALIB')
process.load('Configuration/StandardSequences/MagneticField_cff')
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:run3_data_PromptAnalysis")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(-1))

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("/store/express/Run2023F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/373/710/00000/e2df2f78-b95a-4f33-ae22-add59aa2903f.root"))

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 10000

inTracks = cms.InputTag("ALCARECOSiStripCalMinBias")

process.load('CalibTracker.SiStripCommon.prescaleEvent_cfi')
process.load('CalibTracker.Configuration.Filter_Refit_cff')
## use CalibrationTracks (for clusters) and CalibrationTracksRefit (for tracks)
process.CalibrationTracks.src = inTracks

tracksForCalib = cms.InputTag("CalibrationTracksRefit")

process.prescaleEvent.prescale = 1

process.TkCalSeq = cms.Sequence(process.prescaleEvent*process.MeasurementTrackerEvent*process.trackFilterRefit)

process.load("PhysicsTools.NanoAOD.nano_cff")
process.load("PhysicsTools.NanoAOD.NanoAODEDMEventContent_cff")

## as a test: it should be possible to add tracks fully at configuration level (+ declaring the plugin)
from PhysicsTools.NanoAOD.common_cff import *
## this is equivalent to ShallowTrackProducer as configured for the gain calibration
process.tracksTable = cms.EDProducer("SimpleTrackFlatTableProducer",
        src=tracksForCalib,
        cut=cms.string(""),
        name=cms.string("track"),
        doc=cms.string("SiStripCalMinBias ALCARECO tracks"),
        singleton=cms.bool(False),
        extension=cms.bool(False),
        variables=cms.PSet(
            #chi2=Var("chi2()", float),
            #ndof=Var("ndof()", int),
            chi2ndof=Var("chi2()/ndof", float),
            #charge=Var("charge()", float),
            momentum=Var("p()", float),
            pt=Var("pt()", float),
            #pterr=Var("ptError()", float),
            hitsvalid=Var("numberOfValidHits()", int), ## unsigned?
            #hitslost=Var("numberOfLostHits()", int), ## unsigned?
            #theta=Var("theta()", float),
            #thetaerr=Var("thetaError()", float),
            phi=Var("phi()", float),
            #phierr=Var("phiError()", float),
            eta=Var("eta()", float),
            #etaerr=Var("etaError()", float),
            #dxy=Var("dxy()", float),
            #dxyerr=Var("dxyError()", float),
            #dsz=Var("dsz()", float),
            #dszerr=Var("dszError()", float),
            #qoverp=Var("qoverp()", float),
            #qoverperr=Var("qoverpError()", float),
            #vx=Var("vx()", float),
            #vy=Var("vy()", float),
            #vz=Var("vz()", float),
            algo=Var("algo()", int)
            )
        )
process.load("CalibTracker.SiStripCommon.tkInstLumiTable_cfi")
process.tkInstLumiTable.extension = True
process.load("CalibTracker.SiStripCommon.siStripPositionCorrectionsTable_cfi")
process.load("CalibTracker.SiStripCommon.siStripGainCalibTable_cfi")
process.siStripPositionCorrectionsTable.Tracks = tracksForCalib
process.siStripGainCalibTable.Tracks = tracksForCalib

process.nanoCTPath = cms.Path(process.TkCalSeq *
                              process.nanoMetadata *
                              process.tkInstLumiTable *
                              process.tracksTable *
                              process.siStripPositionCorrectionsTable) #*
                              #process.siStripGainCalibTable)

process.out = cms.OutputModule("NanoAODOutputModule",
                               fileName=cms.untracked.string("CalibTreeMC_nano.root"),
                               outputCommands=process.NANOAODEventContent.outputCommands)

process.end = cms.EndPath(process.out)
