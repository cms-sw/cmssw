import FWCore.ParameterSet.Config as cms

dedxDiscrimProduct = cms.EDProducer("DeDxDiscriminatorProducer",
    tracks                     = cms.InputTag("generalTracks"),
    trajectoryTrackAssociation = cms.InputTag("generalTracks"),

    DiscriminatorMode  = cms.untracked.bool(True),
    MapFile            = cms.string("SingleMuon_DicrimMap.root"),
    Formula            = cms.untracked.uint32(0),

    UseStrip       = cms.bool(True),
    UsePixel       = cms.bool(True),
    MeVperADCStrip = cms.double(3.61e-06*250),
    MeVperADCPixel = cms.double(3.61e-06)
)


doAlldEdXDiscriminators = cms.Sequence(dedxDiscrimProduct)

