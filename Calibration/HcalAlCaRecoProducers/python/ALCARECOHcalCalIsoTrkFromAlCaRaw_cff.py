import FWCore.ParameterSet.Config as cms

#--------------------------------------------------------------------
# AlCaReco filtering for running on dedicated AlCaRaw: AlCaHcalIsoTrk
#--------------------------------------------------------------------

from Calibration.HcalAlCaRecoProducers.alcaHcalIsotrkProducer_cff import alcaHcalIsotrkProducer as _alcaHcalIsotrkProducer
from Calibration.HcalAlCaRecoProducers.alcaHcalIsotrkFilter_cfi import alcaHcalIsotrkFilter as _alcaHcalIsotrkFilter


alcaHcalIsotrkFromAlCaRawProducer = _alcaHcalIsotrkProducer.clone(
    labelTrack = "hltMergedTracksSelector",
    labelVertex = "hltTrimmedPixelVertices",
    labelEBRecHit = ("hltEcalRecHit", "EcalRecHitsEB"),
    labelEERecHit = ("hltEcalRecHit", "EcalRecHitsEE"),
    labelHBHERecHit = "hltHbhereco",
    labelBeamSpot = "hltOnlineBeamSpot",
    labelMuon = "hltIterL3Muons",
    labelTriggerEvent = "hltTriggerSummaryAOD"
)

alcaHcalIsotrkFromAlCaRawFilter = _alcaHcalIsotrkFilter.clone(
    isoTrackLabel = ('alcaHcalIsotrkFromAlCaRawProducer', 'HcalIsoTrack'),
)

seqALCARECOHcalCalIsoTrkFromAlCaRaw = cms.Sequence(alcaHcalIsotrkFromAlCaRawProducer * alcaHcalIsotrkFromAlCaRawFilter)
