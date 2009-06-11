# Validation sequence for (pre)production.
# Based on FastSim Validation and intended to run on RAWDEBUG event content
# i.e. it contains only modules compliant with RAWDEBUG event content (preproduction)

import FWCore.ParameterSet.Config as cms


RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
        mix = cms.PSet(initialSeed = cms.untracked.uint32(12345),
                       engineName = cms.untracked.string('HepJamesRandom')
        ),
        restoreStateLabel = cms.untracked.string("randomEngineStateProducer"),
)

# from fast sim

# Tracking particle module
from SimGeneral.TrackingAnalysis.trackingParticles_cfi import *

# MET
from Validation.RecoMET.METRelValForDQM_cff import *

# tracking
from Validation.TrackingMCTruth.trackingTruthValidation_cfi import *
from Validation.RecoTrack.TrackValidation_cff import *

#muon
from Validation.RecoMuon.muonValidation_cff import *
from Validation.MuonIsolation.MuIsoVal_cff import *

from Validation.RecoMuon.muonValidationHLT_cff import *


# jet
from PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi import * 

from Validation.RecoB.bTagAnalysis_cfi import *
bTagValidation.jetMCSrc = 'IC5byValAlgo'
bTagValidation.etaRanges = cms.vdouble(0.0, 1.1, 2.4)

#hlt
from HLTriggerOffline.Common.HLTValidation_cff import *


# the TrigerSummaryRAW is needed for the following, not available in preprod
hltvalidation.remove(HLTMuonVal)
hltvalidation.remove(egammaValidationSequence)
hltvalidation.remove(HLTJetMETValSeq)
hltvalidation.remove(HLTAlCaVal)


#need to comment the recomuonvalidationhlt_seq in RAWDEBUG (the hltbeamspot is not there)
# the recomuonvalidationhlt_seq is contained in the hltvalidation
# btagvalidatio softelectronbjettags
# either the following two lines or storing as permanent the softelectronbjettags and the hltofflinebeamspot 
from RecoBTag.SoftLepton.softElectronBJetTags_cfi import  *


validationprod = cms.Sequence(trackingTruthValid
                          +tracksValidation
                          +METRelValSequence
                          +recoMuonValidation
                          +muIsoVal_seq
#                          +recoMuonValidationHLT_seq
                          +myPartons
                          +iterativeCone5Flavour
                          +softElectronBJetTags 
                          +bTagValidation
                          +hltvalidation
                          )



