import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from PhysicsTools.PFCandProducer.pfTauSelector_cfi  import *

pfRecoTauProducer.DataType = cms.string('AOD')

ic5PFJetTracksAssociatorAtVertex.jets = 'pfJets'

pfTauSequence = cms.Sequence(
    ic5PFJetTracksAssociatorAtVertex*
    pfRecoTauTagInfoProducer*
    pfRecoTauProducer*
    #    pfRecoTauProducerHighEfficiency*
    pfRecoTauDiscriminationByIsolation*
    #    pfRecoTauDiscriminationHighEfficiency*
    pfTaus
    )


