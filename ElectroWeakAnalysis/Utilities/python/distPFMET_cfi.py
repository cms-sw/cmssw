import FWCore.ParameterSet.Config as cms

# File: PFMET.cff
# Author: R. Cavanaugh
# Date: 28.10.2008
#
# Form uncorrected Missing ET from Particle Flow and store into event as a MET
# product

from RecoMET.METProducers.METSigParams_cfi import *

distpfMet = cms.EDProducer("PFMETProducer",
                       METSignificance_params,
                       src = cms.InputTag("distortedPFCand"),
                       alias = cms.string('PFMET'),
                       globalThreshold = cms.double(0.0),
                       calculateSignificance = cms.bool(True)
                       )
