import FWCore.ParameterSet.Config as cms

pfMuonsFromVertex = cms.EDFilter(
    "IPCutPFCandidateSelector",
    src = cms.InputTag("pfAllMuons"),  # PFCandidate source
    vertices = cms.InputTag("offlinePrimaryVertices"),  # vertices source
    d0Cut = cms.double(0.2),  # transverse IP
    dzCut = cms.double(0.5),  # longitudinal IP
    dtCut = cms.double(-1.0), # time
    d0SigCut = cms.double(99.),  # transverse IP significance
    dzSigCut = cms.double(99.),  # longitudinal IP significance
    dtSigCut = cms.double(-1.0), # time significance
    )
