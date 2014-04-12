import FWCore.ParameterSet.Config as cms
import copy

from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByLeadingTrackFinding_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByLeadingTrackPtCut_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByCharge_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByECALIsolation_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationAgainstElectron_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationAgainstMuon_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByTauPolarization_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByDeltaE_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByInvMass_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByFlightPathSignificance_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByNProngs_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByTrackIsolation_cfi import *

def addCaloDiscriminator(process, tau, name, module):
    module.CaloTauProducer = cms.InputTag(tau+"Producer")
    process.__setattr__(tau+name, module)
    return module

def addCaloDiscriminatorSequence(process, tau):
    lst = []

    lst.append(addCaloDiscriminator(process, tau, "DiscriminationForChargedHiggsByLeadingTrackPtCut",
                                caloRecoTauDiscriminationByLeadingTrackPtCut.clone(
                                   MinPtLeadingObject = cms.double(20.0),
                                   )))

    lst.append(addCaloDiscriminator(process, tau, "DiscriminationByCharge", 
                                caloRecoTauDiscriminationByCharge.clone()))

    # index -1 points to the last element in the list
    lst.append(addCaloDiscriminator(process, tau, "DiscriminationForChargedHiggsByECALIsolation", 
                                caloRecoTauDiscriminationByECALIsolation.clone()))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(tau+'DiscriminationByLeadingTrackFinding')

    lst.append(addCaloDiscriminator(process, tau, "DiscriminationForChargedHiggsAgainstMuon",
                                caloRecoTauDiscriminationAgainstMuon.clone()))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(tau+'DiscriminationByLeadingTrackFinding')

    lst.append(addCaloDiscriminator(process, tau, "DiscriminationByTauPolarization",
                                caloRecoTauDiscriminationByTauPolarization.clone()))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(tau+'DiscriminationByLeadingTrackFinding')

    lst.append(addCaloDiscriminator(process, tau, "DiscriminationByDeltaE",
                                caloRecoTauDiscriminationByDeltaE.clone()))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(tau+'DiscriminationByLeadingTrackFinding')
    
    lst.append(addCaloDiscriminator(process, tau, "DiscriminationByInvMass",
                                caloRecoTauDiscriminationByInvMass.clone()))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(tau+'DiscriminationByLeadingTrackFinding')

    lst.append(addCaloDiscriminator(process, tau, "DiscriminationByFlightPathSignificance",
                                caloRecoTauDiscriminationByFlightPathSignificance.clone()))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(tau+'DiscriminationByLeadingTrackFinding')

    lst.append(addCaloDiscriminator(process, tau, "DiscriminationBy1Prong",
                                caloRecoTauDiscriminationByNProngs.clone(
                                  nProngs = cms.uint32(1)
                                  )))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(tau+'DiscriminationByLeadingTrackFinding')

    lst.append(addCaloDiscriminator(process, tau, "DiscriminationBy3Prongs",
                                caloRecoTauDiscriminationByNProngs.clone(
                                  nProngs = cms.uint32(3)
                                  )))
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(tau+'DiscriminationByLeadingTrackFinding')

    lst.append(addCaloDiscriminator(process, tau, "DiscriminationForChargedHiggsBy3ProngCombined",
                                caloRecoTauDiscriminationByNProngs.clone(
                                  nProngs = cms.uint32(3),
                                  Prediscriminants = cms.PSet(
	                               BooleanOperator = cms.string("and"),
	                               leadTrack = cms.PSet(
	                                   Producer = cms.InputTag(tau+'DiscriminationByLeadingTrackFinding'),
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
    lst[-1].Prediscriminants.leadTrack.Producer = cms.InputTag(tau+'DiscriminationByLeadingTrackFinding')

    lst.append(addCaloDiscriminator(process, tau, "DiscriminationForChargedHiggsBy1or3Prongs",
                                caloRecoTauDiscriminationByLeadingTrackFinding.clone(
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
    lst.append(addCaloDiscriminator(process, tau, "DiscriminationForChargedHiggs",
       			        caloRecoTauDiscriminationByTrackIsolation.clone(
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
	 			            Producer = cms.InputTag(tau+'DiscriminationAgainstElectron'),
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

def addCaloTauDiscriminationSequenceForChargedHiggs(process):
    process.CaloTauDiscriminationSequenceForChargedHiggs = cms.Sequence(
	addCaloDiscriminatorSequence(process, "caloRecoTau")
    )

    return process.CaloTauDiscriminationSequenceForChargedHiggs
