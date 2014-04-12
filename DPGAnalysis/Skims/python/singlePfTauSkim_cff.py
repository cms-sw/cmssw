import FWCore.ParameterSet.Config as cms
import copy

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolationUsingLeadingPion_cfi import *
from RecoTauTag.RecoTau.TauDiscriminatorTools import *
#Need this next one to put the transientTrackRecord in and avoid crashes 
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *

from RecoTauTag.Configuration.ShrinkingConePFTaus_cff import *
looseShrikingConePFTaus = copy.deepcopy(shrinkingConePFTauProducer)
looseShrikingConePFTaus.LeadPFCand_minPt = cms.double(3.0)

thePFTauDiscByLeadTrkFinding = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackFinding)
thePFTauDiscByLeadTrkFinding.PFTauProducer = cms.InputTag('looseShrikingConePFTaus')

thePFTauDiscByIsolation = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
thePFTauDiscByIsolation.PFTauProducer = cms.InputTag('looseShrikingConePFTaus')
thePFTauDiscByIsolation.Prediscriminants = cms.PSet(
      BooleanOperator = cms.string("and"),
      leadTrack = cms.PSet(
      Producer = cms.InputTag('thePFTauDiscByLeadTrkFinding'),
      cut = cms.double(0.5)
      )
)

PFTausSelected = cms.EDFilter("PFTauSelector",
    src = cms.InputTag("looseShrikingConePFTaus"),
    discriminators = cms.VPSet(
	cms.PSet( discriminator=cms.InputTag("thePFTauDiscByIsolation"),
		   selectionCut=cms.double(0.5)
	)
    ),
    cut = cms.string('et > 15. && abs(eta) < 2.5') 
)

PFTauSkimmed = cms.EDFilter("CandViewCountFilter",
  src = cms.InputTag('PFTausSelected'),
  minNumber = cms.uint32(1)
)

singlePfTauPt15QualitySeq = cms.Sequence(
    looseShrikingConePFTaus+thePFTauDiscByLeadTrkFinding+thePFTauDiscByIsolation+PFTausSelected+PFTauSkimmed
    )
