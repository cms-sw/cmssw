import FWCore.ParameterSet.Config as cms

from RecoJets.FFTJetProducers.fftjetcommon_cfi import *

fftjetPfPileupCleaner = cms.EDProducer(
    "FFTJetPFPileupCleaner",
    #
    # Label for the input collection of PFCandidate objects
    PFCandidates = cms.InputTag("particleFlow"),
    #
    # Label for the collection of primary vertices
    Vertices = cms.InputTag("offlinePrimaryVertices"),
    #
    # Info about fake primary vertices
    useFakePrimaryVertex = cms.bool(False),
    FakePrimaryVertices = cms.InputTag("vertexadder", "FFTJetFudgedVertices"),
    #
    # Find the closest vertex even if the track is not associated
    # with any good vertex?
    checkClosestZVertex = cms.bool(True),
    #
    # Associate all tracks neigboring the main vertex with it?
    # This switch is meaningful only if "checkClosestZVertex" it true.
    keepIfPVneighbor = cms.bool(True),
    #
    # Remove the objects associated with the main primary vertex?
    removeMainVertex = cms.bool(False),
    #
    # Remove the objects not associated with any primary vertex?
    removeUnassociated = cms.bool(False),
    #
    # Overall flag to invert the decision
    reverseRemovalDecision = cms.bool(False),
    #
    # Various removal flags by object type. See PFCandidate header
    # for object type details.
    remove_X = cms.bool(False),
    remove_h = cms.bool(True),
    remove_e = cms.bool(True),
    remove_mu = cms.bool(True),
    remove_gamma = cms.bool(False),
    remove_h0 = cms.bool(False),
    remove_h_HF = cms.bool(False),
    remove_egamma_HF = cms.bool(False),
    #
    # Minimum and maximum allowed eta
    etaMin = cms.double(-fftjet_standard_eta_range),
    etaMax = cms.double(fftjet_standard_eta_range),
    #
    # Vertex quality cuts
    vertexNdofCut = cms.double(4.0),
    vertexZmaxCut = cms.double(24.0)
)
