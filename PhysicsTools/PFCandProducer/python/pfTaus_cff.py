# TODO: change all names consisting of 'allLayer0Taus' for less misleading ones
import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from RecoTauTag.TauTagTools.PFTauSelector_cfi  import pfTauSelector


fixedConePFTauProducer.DataType = cms.string('AOD')
ic5PFJetTracksAssociatorAtVertex.jets = 'pfJets'

# Clone tau discriminants to avoid further problems with PAT
allLayer0TausDiscriminationByLeadTrackPt = fixedConePFTauDiscriminationByLeadingTrackPtCut.clone()
allLayer0TausDiscriminationByIsolation = fixedConePFTauDiscriminationByIsolation.clone()

allLayer0Taus = pfTauSelector.clone()
allLayer0Taus.src = cms.InputTag("fixedConePFTauProducer")
allLayer0Taus.discriminators = cms.VPSet(
      #cms.PSet( discriminator=cms.InputTag("fixedConePFTauDiscriminationByIsolation"),selectionCut=cms.double(0.5))
      cms.PSet( discriminator=cms.InputTag("allLayer0TausDiscriminationByLeadTrackPt"),selectionCut=cms.double(0.5) ),
      cms.PSet( discriminator=cms.InputTag("allLayer0TausDiscriminationByIsolation"),selectionCut=cms.double(0.5) )
   )

pfRecoTauTagInfoProducer.PFCandidateProducer = 'pfNoElectron'


pfTauSequence = cms.Sequence(
    ic5PFJetTracksAssociatorAtVertex + 
    pfRecoTauTagInfoProducer + 
    fixedConePFTauProducer + 
    #fixedConePFTauDiscriminationByIsolation +
    allLayer0TausDiscriminationByIsolation +
    allLayer0TausDiscriminationByLeadTrackPt +
    allLayer0Taus 
    )


