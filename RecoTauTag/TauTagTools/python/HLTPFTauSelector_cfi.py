import FWCore.ParameterSet.Config as cms
from RecoTauTag.TauTagTools.PFTauSelector_cfi.py import pfTauSelector
hltPFTauSelector = pfTauSelector.clone(
    src = cms.InputTag("fixedConePFTauProducer"),
    discriminators = cms.VPSet(
        cms.PSet( discriminator=cms.InputTag("fixedConePFTauDiscriminationByIsolation"),selectionCut=cms.double(0.5))
    ),
    cut = cms.string("pt > 0"),
)


