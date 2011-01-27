import FWCore.ParameterSet.Config as cms

# File: PFMET.cff
# Author: R. Cavanaugh
# Date: 28.10.2008
#
# Form uncorrected Missing ET from Particle Flow and store into event as a MET
# product

from RecoMET.METProducers.METSigParams_cfi import *

pfClusterMet = cms.EDProducer("METProducer",
                              METSignificance_params,
                              src = cms.InputTag("pfClusterRefsForJets"),
                              METType = cms.string('PFClusterMET'),
                              alias = cms.string('PFClusterMET'),
                              noHF = cms.bool(False),
                              globalThreshold = cms.double(0.0),
                              InputType = cms.string('RecoPFClusterRefCandidateCollection'),
                              calculateSignificance = cms.bool(False)
                              )
