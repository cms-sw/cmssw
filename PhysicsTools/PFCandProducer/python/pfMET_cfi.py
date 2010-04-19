import FWCore.ParameterSet.Config as cms

from RecoMET.METProducers.METSigParams_cfi import *

pfMET = cms.EDProducer("METProducer",
                       
                       METSignificance_params,
                       src = cms.InputTag("particleFlow"),
                       METType = cms.string('PFMET'),
                       alias = cms.string('PFMET'),
                       noHF = cms.bool(False),
                       globalThreshold = cms.double(0.0),
                       InputType = cms.string('PFCandidateCollection'),
                       calculateSignificance = cms.bool(True)
                       )

