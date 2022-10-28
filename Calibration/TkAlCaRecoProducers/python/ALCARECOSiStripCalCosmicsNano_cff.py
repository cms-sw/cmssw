import FWCore.ParameterSet.Config as cms

from Calibration.TkAlCaRecoProducers.ALCARECOSiStripCalCosmics_cff import ALCARECOSiStripCalCosmics
from CalibTracker.SiStripCommon.prescaleEvent_cfi import prescaleEvent
from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter

ALCARECOSiStripCalCosmicsNanoPrescale = prescaleEvent.clone(prescale=1)

ALCARECOSiStripCalCosmicsNanoHLT = triggerResultsFilter.clone(
        triggerConditions=cms.vstring("HLT_L1SingleMuCosmics_v*"),
        hltResults=cms.InputTag("TriggerResults", "", "HLT"),
        l1tResults=cms.InputTag(""),
        throw=cms.bool(False)
        )

# Select only events where tracker had HV on (according to DCS bit information)
# AND respective partition is in the run (according to FED information)
import CalibTracker.SiStripCommon.SiStripDCSFilter_cfi
DCSStatusForSiStripCalCosmicsNano = CalibTracker.SiStripCommon.SiStripDCSFilter_cfi.siStripDCSFilter.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
    )

from CalibTracker.Configuration.Filter_Refit_cff import CalibrationTracks, CalibrationTracksRefit, MeasurementTrackerEvent, offlineBeamSpot

ALCARECOSiStripCalCosmicsNanoCalibTracks = CalibrationTracks.clone(src=cms.InputTag("ALCARECOSiStripCalCosmics"))
ALCARECOSiStripCalCosmicsNanoCalibTracksRefit = CalibrationTracksRefit.clone(
        src=cms.InputTag("ALCARECOSiStripCalCosmicsNanoCalibTracks")
        )

ALCARECOSiStripCalCosmicsNanoTkCalSeq = cms.Sequence(
        ALCARECOSiStripCalCosmicsNanoPrescale*
        ALCARECOSiStripCalCosmicsNanoHLT*
        DCSStatusForSiStripCalCosmicsNano*
        ALCARECOSiStripCalCosmicsNanoCalibTracks,
        cms.Task(MeasurementTrackerEvent),
        cms.Task(offlineBeamSpot),
        cms.Task(ALCARECOSiStripCalCosmicsNanoCalibTracksRefit)
        )

from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_cff import nanoMetadata
from CalibTracker.SiStripCommon.siStripPositionCorrectionsTable_cfi import siStripPositionCorrectionsTable
from CalibTracker.SiStripCommon.siStripLorentzAngleRunInfoTable_cfi import siStripLorentzAngleRunInfoTable

ALCARECOSiStripCalCosmicsNanoTracksTable = cms.EDProducer("SimpleTrackFlatTableProducer",
        src=cms.InputTag("ALCARECOSiStripCalCosmicsNanoCalibTracksRefit"),
        cut=cms.string(""),
        name=cms.string("track"),
        doc=cms.string("SiStripCalCosmics ALCARECO tracks"),
        singleton=cms.bool(False),
        extension=cms.bool(False),
        variables=cms.PSet(
            chi2ndof=Var("chi2()/ndof", float),
            pt=Var("pt()", float),
            hitsvalid=Var("numberOfValidHits()", int), ## unsigned?
            phi=Var("phi()", float),
            eta=Var("eta()", float),
            )
        )

ALCARECOSiStripCalCosmicsNanoMeasTable = siStripPositionCorrectionsTable.clone(
        Tracks=cms.InputTag("ALCARECOSiStripCalCosmicsNanoCalibTracksRefit"))

ALCARECOSiStripCalCosmicsNanoTables = cms.Task(
        nanoMetadata,
        ALCARECOSiStripCalCosmicsNanoTracksTable,
        ALCARECOSiStripCalCosmicsNanoMeasTable,
        siStripLorentzAngleRunInfoTable
        )

seqALCARECOSiStripCalCosmicsNano = cms.Sequence(ALCARECOSiStripCalCosmicsNanoTkCalSeq, ALCARECOSiStripCalCosmicsNanoTables)
