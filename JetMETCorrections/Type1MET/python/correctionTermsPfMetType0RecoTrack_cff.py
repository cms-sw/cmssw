import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from JetMETCorrections.Type1MET.pfMETCorrections_cff import *

##____________________________________________________________________________||
corrPfMetType0RecoTrack = cms.EDProducer(
    "ScaleCorrMETData",
    src = cms.InputTag('pfchsMETcorr', 'type0'),
    scaleFactor = cms.double(1 - 0.6)
    )

##____________________________________________________________________________||
corrPfMetType0RecoTrackForType2 = cms.EDProducer(
    "ScaleCorrMETData",
    src = cms.InputTag('corrPfMetType0RecoTrack'),
    scaleFactor = cms.double(1.4)
    )

##____________________________________________________________________________||
correctionTermsPfMetType0RecoTrack = cms.Sequence(
    pfchsMETcorr +
    corrPfMetType0RecoTrack +
    corrPfMetType0RecoTrackForType2
    )

##____________________________________________________________________________||
