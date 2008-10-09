import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from PhysicsTools.PFCandProducer.pfTauSelector_cfi  import *

pfRecoTauProducer.DataType = cms.string('AOD')
myic5PFJetTracksAssociatorAtVertex = ic5PFJetTracksAssociatorAtVertex.clone()

myic5PFJetTracksAssociatorAtVertex.jets = 'pfJets'
pfRecoTauTagInfoProducer.PFJetTracksAssociatorProducer = 'myic5PFJetTracksAssociatorAtVertex'

pfTauSequence = cms.Sequence(
    myic5PFJetTracksAssociatorAtVertex + 
    pfRecoTauTagInfoProducer + 
    pfRecoTauProducer + 
    #    pfRecoTauProducerHighEfficiency + 
    pfRecoTauDiscriminationByIsolation + 
    #    pfRecoTauDiscriminationHighEfficiency + 
    pfTaus
    )


