import FWCore.ParameterSet.Config as cms
# $Id: RecoMET_cff.py,v 1.19 2012/11/06 02:33:52 sakuma Exp $

##____________________________________________________________________________||
pfCandidatesForTrackMet = cms.EDProducer(
    "PFCandidatesForTrackMETProducer",
    PFCollectionLabel = cms.InputTag("particleFlow"),
    PVCollectionLabel = cms.InputTag("offlinePrimaryVertices"),
    dzCut = cms.double(0.2),
    neutralEtThreshold = cms.double(-1.0)
    )

##____________________________________________________________________________||
trackMet = cms.EDProducer(
    "METProducer",
    src = cms.InputTag("pfCandidatesForTrackMet"),
    METType = cms.string('PFMET'),
    alias = cms.string('PFMET'),
    globalThreshold = cms.double(0.0),
    InputType = cms.string('PFCandidateCollection'),
    calculateSignificance = cms.bool(False),
    )

##____________________________________________________________________________||
