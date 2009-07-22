import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from RecoTauTag.TauTagTools.PFTauSelector_cfi  import pfTauSelector

shrinkingConePFTauProducer.DataType = cms.string('AOD')
ic5PFJetTracksAssociatorAtVertex.jets = 'pfJets'

allLayer0Taus = pfTauSelector.clone()
allLayer0Taus.src = cms.InputTag("shrinkingConePFTauProducer")
allLayer0Taus.discriminators = cms.VPSet(
      cms.PSet( discriminator=cms.InputTag("shrinkingConePFTauDiscriminationByIsolation"),selectionCut=cms.double(0.5))
   )

pfRecoTauTagInfoProducer.PFCandidateProducer = 'pfNoMuon'


pfTauSequence = cms.Sequence(
    ic5PFJetTracksAssociatorAtVertex + 
    pfRecoTauTagInfoProducer + 
    shrinkingConePFTauProducer + 
    shrinkingConePFTauDiscriminationByIsolation + 
    allLayer0Taus 

    )


