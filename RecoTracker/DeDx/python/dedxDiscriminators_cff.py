import FWCore.ParameterSet.Config as cms

dedxDiscrimProductProd = cms.EDProducer("DeDxDiscriminatorProducer",
    tracks                     = cms.InputTag("generalTracks"),
    trajectoryTrackAssociation = cms.InputTag("generalTracks"),

    DiscriminatorMode  = cms.untracked.bool(True),
    MapFile            = cms.string("SingleMuon_DicrimMap.root"),
    Formula            = cms.untracked.uint32(0),

    UseStrip           = cms.bool(True),
    UsePixel           = cms.bool(True),
    MeVperADCStrip     = cms.double(3.61e-06*250),
    MeVperADCPixel     = cms.double(3.61e-06)
)

dedxDiscrimProductBTag = cms.EDProducer("DeDxDiscriminatorProducer",
    tracks                     = cms.InputTag("generalTracks"),
    trajectoryTrackAssociation = cms.InputTag("generalTracks"),

    DiscriminatorMode  = cms.untracked.bool(True),
    MapFile            = cms.string("SingleMuon_DicrimMap.root"),
    Formula            = cms.untracked.uint32(1),

    UseStrip           = cms.bool(True),
    UsePixel           = cms.bool(True),
    MeVperADCStrip     = cms.double(3.61e-06*250),
    MeVperADCPixel     = cms.double(3.61e-06)
)

dedxDiscrimProductSmi = cms.EDProducer("DeDxDiscriminatorProducer",
    tracks                     = cms.InputTag("generalTracks"),
    trajectoryTrackAssociation = cms.InputTag("generalTracks"),

    DiscriminatorMode  = cms.untracked.bool(True),
    MapFile            = cms.string("SingleMuon_DicrimMap.root"),
    Formula            = cms.untracked.uint32(2),

    UseStrip           = cms.bool(True),
    UsePixel           = cms.bool(True),
    MeVperADCStrip     = cms.double(3.61e-06*250),
    MeVperADCPixel     = cms.double(3.61e-06)
)

dedxDiscrimProductASmi = cms.EDProducer("DeDxDiscriminatorProducer",
    tracks                     = cms.InputTag("generalTracks"),
    trajectoryTrackAssociation = cms.InputTag("generalTracks"),

    DiscriminatorMode  = cms.untracked.bool(True),
    MapFile            = cms.string("SingleMuon_DicrimMap.root"),
    Formula            = cms.untracked.uint32(3),

    UseStrip           = cms.bool(True),
    UsePixel           = cms.bool(True),
    MeVperADCStrip     = cms.double(3.61e-06*250),
    MeVperADCPixel     = cms.double(3.61e-06)
)

doAlldEdXDiscriminators = cms.Sequence(dedxDiscrimProductProd * dedxDiscrimProductBTag * dedxDiscrimProductSmi * dedxDiscrimProductASmi)

