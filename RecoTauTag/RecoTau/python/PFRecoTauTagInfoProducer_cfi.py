import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
import RecoTauTag.RecoTau.pfRecoTauTagInfoProducer_cfi as _mod

pfRecoTauTagInfoProducer = _mod.pfRecoTauTagInfoProducer.clone(
    # These values set the minimum pt quality requirements
    # for the various constituent types
    ChargedHadrCand_tkminPt     = 0.5,  # charged PF objects
    tkminPt                     = 0.5,  # track (non-PF) objects
    NeutrHadrCand_HcalclusMinEt = 1.0,  # PF neutral hadrons (HCAL)
    GammaCand_EcalclusMinEt     = 1.0,  # PF gamma candidates (ECAL)

    # The size of the delta R cone used to collect objects from the jet
    ChargedHadrCand_AssociationCone   = 0.8,
    PVProducer                        = PFTauQualityCuts.primaryVertexSrc,

    # Quality cuts for tracks (non-PF, from JetTracksAssociator)
    tkminTrackerHitsn = 3,
    tkmaxChi2         = 100.0,
    tkPVmaxDZ         = 0.2, ##considered if UsePVconstraint is true
    tkminPixelHitsn   = 0,
    tkmaxipt          = 0.03,

    # Quality cuts for PFCharged Hadron candidates (taken from their underlying recTrack)
    ChargedHadrCand_tkminTrackerHitsn = 3,
    ChargedHadrCand_tkmaxChi2         = 100.0,
    ChargedHadrCand_tkmaxipt          = 0.03,
    ChargedHadrCand_tkminPixelHitsn   = 0,
    ChargedHadrCand_tkPVmaxDZ         = 0.2, ##considered if UsePVconstraint is true

    # Smear vertex
    smearedPVsigmaY               = 0.0015,
    smearedPVsigmaX               = 0.0015,
    smearedPVsigmaZ               = 0.005,
)

# PF TauTag info seeded from the Inside-Out jet producer
pfRecoTauTagInfoProducerInsideOut = pfRecoTauTagInfoProducer.clone(
    PFJetTracksAssociatorProducer   = 'insideOutJetTracksAssociatorAtVertex',
    ChargedHadrCand_AssociationCone = 1.0
)
