import FWCore.ParameterSet.Config as cms

hltHpsSelectedPFTausTrackFinding = cms.EDFilter("PFTauSelector",
    cut = cms.string('pt > 0'),
    discriminatorContainers = cms.VPSet(),
    discriminators = cms.VPSet(cms.PSet(
        discriminator = cms.InputTag("hltHpsPFTauTrackFindingDiscriminator"),
        selectionCut = cms.double(0.5)
    )),
    src = cms.InputTag("hltHpsPFTauProducer")
)
