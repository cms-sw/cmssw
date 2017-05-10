import FWCore.ParameterSet.Config as cms

PFTau3ProngReco = cms.EDProducer("PFTau3ProngReco",
                                 PFTauTag =  cms.InputTag("hpsPFTauProducer"),
                                 PFTauTIPTag = cms.InputTag("hpsPFTauTransverseImpactParameters"),
                                 Algorithm = cms.int32(0),
                                 discriminators = cms.VPSet(cms.PSet(discriminator = cms.InputTag('hpsPFTauDiscriminationByDecayModeFinding'),selectionCut = cms.double(0.5))),
                                 cut = cms.string("pt > 18.0 & abs(eta)<2.3")
                                 )
