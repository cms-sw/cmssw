import FWCore.ParameterSet.Config as cms
import copy

from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import *
hplusTrackQualityCuts = PFTauQualityCuts.clone()
hplusTrackQualityCuts.maxTrackChi2 = cms.double(10.)
hplusTrackQualityCuts.minTrackHits = cms.uint32(8)

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackPtCut_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByCharge_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByECALIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByTauPolarization_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByDeltaE_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByInvMass_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByFlightPathSignificance_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByNProngs_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolation_cfi import *

def addDiscriminator(process, tau, name, module):
    producerPostfix = ""
    if tau != "hpsTancTaus":
        producerPostfix = "Producer"
    module.PFTauProducer = cms.InputTag(tau+producerPostfix)
    process.__setattr__(tau+name, module)
    return module

def addDiscriminatorSequence(process, tau):
    leadingTrackFinding = tau+"DiscriminationByLeadingTrackFinding"
    if tau == "hpsPFTau":
        leadingTrackFinding = tau+"DiscriminationByDecayModeFinding"
    
    lst = []

    lst.append(addDiscriminator(process, tau, "DiscriminationForChargedHiggsByLeadingTrackPtCut",
                                pfRecoTauDiscriminationByLeadingTrackPtCut.clone(
                                   MinPtLeadingObject = cms.double(20.0),
                                   qualityCuts = hplusTrackQualityCuts
                                   )))

    lst.append(addDiscriminator(process, tau, "DiscriminationByCharge", 
                                pfRecoTauDiscriminationByCharge.clone()))

    # index -1 points to the last element in the list
    lst.append(addDiscriminator(process, tau, "DiscriminationForChargedHiggsByECALIsolation", 
                                pfRecoTauDiscriminationByECALIsolation.clone()))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(leadingTrackFinding)

    lst.append(addDiscriminator(process, tau, "DiscriminationForChargedHiggsAgainstElectron",
                                pfRecoTauDiscriminationAgainstElectron.clone()))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(leadingTrackFinding)

    lst.append(addDiscriminator(process, tau, "DiscriminationForChargedHiggsAgainstMuon",
                                pfRecoTauDiscriminationAgainstMuon.clone()))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(leadingTrackFinding)

    lst.append(addDiscriminator(process, tau, "DiscriminationByTauPolarization",
                                pfRecoTauDiscriminationByTauPolarization.clone()))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(leadingTrackFinding)

    lst.append(addDiscriminator(process, tau, "DiscriminationByDeltaE",
                                pfRecoTauDiscriminationByDeltaE.clone()))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(leadingTrackFinding)
    
    lst.append(addDiscriminator(process, tau, "DiscriminationByInvMass",
                                pfRecoTauDiscriminationByInvMass.clone()))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(leadingTrackFinding)

    lst.append(addDiscriminator(process, tau, "DiscriminationByFlightPathSignificance",
                                pfRecoTauDiscriminationByFlightPathSignificance.clone()))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(leadingTrackFinding)

    lst.append(addDiscriminator(process, tau, "DiscriminationBy1Prong",
                                pfRecoTauDiscriminationByNProngs.clone(
                                  nProngs = cms.uint32(1)
                                  )))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(leadingTrackFinding)

    lst.append(addDiscriminator(process, tau, "DiscriminationBy3Prongs",
                                pfRecoTauDiscriminationByNProngs.clone(
                                  nProngs = cms.uint32(3)
                                  )))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(leadingTrackFinding)

    lst.append(addDiscriminator(process, tau, "DiscriminationForChargedHiggsBy3ProngCombined",
                                pfRecoTauDiscriminationByNProngs.clone(
                                  nProngs = cms.uint32(3),
                                  Prediscriminants = cms.PSet(
	                               BooleanOperator = cms.string("and"),
	                               leadTrack = cms.PSet(
	                                   Producer = cms.InputTag(leadingTrackFinding),
	                                   cut = cms.double(0.5)
	                               ),
	                               deltaE = cms.PSet(
	                                   Producer = cms.InputTag(tau+'DiscriminationByDeltaE'),
	                                   cut = cms.double(0.5)
	                               ),
	                               invMass = cms.PSet(
	                                   Producer = cms.InputTag(tau+'DiscriminationByInvMass'),
	                                   cut = cms.double(0.5)
	                               ),
	                               flightPathSig = cms.PSet(
	                                   Producer = cms.InputTag(tau+'DiscriminationByFlightPathSignificance'),
	                                   cut = cms.double(0.5)
	                               )
	                          )
                                )))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(leadingTrackFinding)

    lst.append(addDiscriminator(process, tau, "DiscriminationForChargedHiggsBy1or3Prongs",
                                pfRecoTauDiscriminationByLeadingTrackFinding.clone(
	 			    Prediscriminants = cms.PSet(
	 			        BooleanOperator = cms.string("or"),
	 			        oneProng = cms.PSet(
	 			            Producer = cms.InputTag(tau+'DiscriminationBy1Prong'),
	 			            cut = cms.double(0.5)
	 			        ),
	 			        threeProng = cms.PSet(
	 			            Producer = cms.InputTag(tau+'DiscriminationForChargedHiggsBy3ProngCombined'),
	 			            cut = cms.double(0.5)
	 			        )
	 			    )
	 			)))
    lst.append(addDiscriminator(process, tau, "DiscriminationForChargedHiggs",
       			        pfRecoTauDiscriminationByTrackIsolation.clone(
	                             Prediscriminants = cms.PSet(
	 			        BooleanOperator = cms.string("and"),
	 			        leadingTrack = cms.PSet(
	 			            Producer = cms.InputTag(tau+'DiscriminationForChargedHiggsByLeadingTrackPtCut'),
	 			            cut = cms.double(0.5)
	 			        ),
	 			        charge = cms.PSet(
	 			            Producer = cms.InputTag(tau+'DiscriminationByCharge'),
	 			            cut = cms.double(0.5)
	 			        ),
	 			        ecalIsolation = cms.PSet(
	 			            Producer = cms.InputTag(tau+'DiscriminationForChargedHiggsByECALIsolation'),
	 			            cut = cms.double(0.5)
	 			        ),
	 			        electronVeto = cms.PSet(
	 			            Producer = cms.InputTag(tau+'DiscriminationForChargedHiggsAgainstElectron'),
	 			            cut = cms.double(0.5)
	 			        ),
	 			        polarization = cms.PSet(
	 			            Producer = cms.InputTag(tau+'DiscriminationByTauPolarization'),
	 			            cut = cms.double(0.5)
	 			        ),
	 			        prongs = cms.PSet(
	 			            Producer = cms.InputTag(tau+'DiscriminationForChargedHiggsBy1or3Prongs'),
	 			            cut = cms.double(0.5)
	 			        )
	 			    )
	 			)))

    sequence = cms.Sequence()
    for m in lst:
        sequence *= m

    process.__setattr__(tau+"HplusDiscriminationSequence", sequence)
    return sequence

def addPFTauDiscriminationSequenceForChargedHiggs(process, tauAlgos=["shrinkingConePFTau"]):
    process.PFTauDiscriminationSequenceForChargedHiggs = cms.Sequence()
    for algo in tauAlgos:
        process.PFTauDiscriminationSequenceForChargedHiggs *= addDiscriminatorSequence(process, algo)

    return process.PFTauDiscriminationSequenceForChargedHiggs
