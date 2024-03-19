import FWCore.ParameterSet.Config as cms

hltHpsSelectedPFTausTrackPt1MediumChargedIsolation = cms.EDFilter("PFTauSelector",
    cut = cms.string('pt > 0'),
    discriminatorContainers = cms.VPSet(),
    discriminators = cms.VPSet(cms.PSet(
        discriminator = cms.InputTag("hltHpsPFTauMediumAbsOrRelChargedIsolationDiscriminator"),
        selectionCut = cms.double(0.5)
    )),
    src = cms.InputTag("hltHpsPFTauProducer")
)
