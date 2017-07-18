import FWCore.ParameterSet.Config as cms

ctppsLocalTrackLiteProducer = cms.EDProducer("CTPPSLocalTrackLiteProducer",
    tagSiStripTrack = cms.InputTag("totemRPLocalTrackFitter"),
    tagDiamondTrack = cms.InputTag("ctppsDiamondLocalTracks"),

    # disable the module by default
    doNothing = cms.bool(True)
)

# enable the module for CTPPS era(s)
from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
ctpps_2016.toModify(ctppsLocalTrackLiteProducer, doNothing=cms.bool(False))
