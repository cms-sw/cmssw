import FWCore.ParameterSet.Config as cms
import copy
from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts

pfRecoTauTagInfoProducer = cms.EDProducer("PFRecoTauTagInfoProducer",

    # These values set the minimum pt quality requirements
    #  for the various constituent types
    ChargedHadrCand_tkminPt    = cms.double(0.5),  # charged PF objects
    tkminPt                    = cms.double(0.5),  # track (non-PF) objects
    NeutrHadrCand_HcalclusMinEt = cms.double(1.0),  # PF neutral hadrons (HCAL)
    GammaCand_EcalclusMinEt     = cms.double(1.0),  # PF gamma candidates (ECAL)

    # The size of the delta R cone used to collect objects from the jet
    ChargedHadrCand_AssociationCone   = cms.double(0.8),

    PVProducer                    = PFTauQualityCuts.primaryVertexSrc,
    UsePVconstraint               = cms.bool(True),
    PFCandidateProducer           = cms.InputTag('particleFlow'),
    PFJetTracksAssociatorProducer = cms.InputTag('ak4PFJetTracksAssociatorAtVertex'),

    # Quality cuts for tracks (non-PF, from JetTracksAssociator)
    tkminTrackerHitsn = cms.int32(3),
    tkmaxChi2         = cms.double(100.0),
    tkPVmaxDZ         = cms.double(0.2), ##considered if UsePVconstraint is true
    tkminPixelHitsn   = cms.int32(0),
    tkmaxipt          = cms.double(0.03),

    # Quality cuts for PFCharged Hadron candidates (taken from their underlying recTrack)
    ChargedHadrCand_tkminTrackerHitsn = cms.int32(3),
    ChargedHadrCand_tkmaxChi2         = cms.double(100.0),
    ChargedHadrCand_tkmaxipt          = cms.double(0.03),
    ChargedHadrCand_tkminPixelHitsn   = cms.int32(0),
    ChargedHadrCand_tkPVmaxDZ         = cms.double(0.2), ##considered if UsePVconstraint is true

    # Smear vertex
    smearedPVsigmaY               = cms.double(0.0015),
    smearedPVsigmaX               = cms.double(0.0015),
    smearedPVsigmaZ               = cms.double(0.005),
)

# PF TauTag info seeded from the Inside-Out jet producer
pfRecoTauTagInfoProducerInsideOut                                 = copy.deepcopy(pfRecoTauTagInfoProducer)
pfRecoTauTagInfoProducerInsideOut.PFJetTracksAssociatorProducer   = cms.InputTag('insideOutJetTracksAssociatorAtVertex')
pfRecoTauTagInfoProducerInsideOut.ChargedHadrCand_AssociationCone = cms.double(1.0)

