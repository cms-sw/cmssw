import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from RecoTauTag.TauTagTools.PFTauSelector_cfi  import pfTauSelector

pfRecoTauProducerHighEfficiency.DataType = cms.string('AOD')
ic5PFJetTracksAssociatorAtVertex.jets = 'pfJets'

allLayer0Taus = pfTauSelector.clone()
allLayer0Taus.src = cms.InputTag("pfRecoTauProducerHighEfficiency")
allLayer0Taus.discriminators = cms.VPSet(
      cms.PSet( discriminator=cms.InputTag("pfRecoTauDiscriminationByIsolationHighEfficiency"),selectionCut=cms.double(0.5))
   )

pfRecoTauTagInfoProducer.PFCandidateProducer = 'pfNoMuonsNoPileUp:PFCandidates'


pfTauSequence = cms.Sequence(
    ic5PFJetTracksAssociatorAtVertex + 
    pfRecoTauTagInfoProducer + 
    pfRecoTauProducerHighEfficiency + 
    pfRecoTauDiscriminationByIsolationHighEfficiency + 
    allLayer0Taus
    )


