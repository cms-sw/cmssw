import FWCore.ParameterSet.Config as cms


pfCandidatesForTrackMet = cms.EDProducer("PFCandidatesForTrackMETProducer",
                                         PFCollectionLabel = cms.InputTag("particleFlow"),
                                         PVCollectionLabel = cms.InputTag("offlinePrimaryVertices"),
                                         dzCut = cms.double(0.2),
                                         neutralEtThreshold = cms.double(-1.0)
                                         )

from RecoMET.METProducers.METSigParams_cfi import *
trackMet = cms.EDProducer("METProducer",
                          METSignificance_params,
                          src = cms.InputTag("pfCandidatesForTrackMet"),
                          METType = cms.string('PFMET'),
                          alias = cms.string('PFMET'),
                          noHF = cms.bool(False),
                          globalThreshold = cms.double(0.0),
                          InputType = cms.string('PFCandidateCollection'),
                          calculateSignificance = cms.bool(True),
                          jets = cms.InputTag("ak5PFJets") #used for significance calculation
                          )
