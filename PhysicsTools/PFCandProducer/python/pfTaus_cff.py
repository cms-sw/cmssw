import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from RecoTauTag.TauTagTools.PFTauSelector_cfi  import pfTauSelector

pfRecoTauProducer.DataType = cms.string('AOD')
myic5PFJetTracksAssociatorAtVertex = ic5PFJetTracksAssociatorAtVertex.clone()

myic5PFJetTracksAssociatorAtVertex.jets = 'pfJets'
pfRecoTauTagInfoProducer.PFJetTracksAssociatorProducer = 'myic5PFJetTracksAssociatorAtVertex'

pfTaus = pfTauSelector.clone()
pfTaus.discriminators = cms.VPSet(
      cms.PSet( discriminator=cms.InputTag("pfRecoTauDiscriminationByIsolation"),selectionCut=cms.double(0.5))
   )


pfTauSequence = cms.Sequence(
    myic5PFJetTracksAssociatorAtVertex + 
    pfRecoTauTagInfoProducer + 
    pfRecoTauProducer + 
    #    pfRecoTauProducerHighEfficiency + 
    pfRecoTauDiscriminationByIsolation + 
    #    pfRecoTauDiscriminationHighEfficiency + 
    pfTaus
    )


