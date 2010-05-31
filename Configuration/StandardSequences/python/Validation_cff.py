import FWCore.ParameterSet.Config as cms


RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
        mix = cms.PSet(initialSeed = cms.untracked.uint32(12345),
                       engineName = cms.untracked.string('HepJamesRandom')
        ),
        restoreStateLabel = cms.untracked.string("randomEngineStateProducer"),
)

from Validation.GlobalDigis.globaldigis_analyze_cfi import *
from Validation.GlobalRecHits.globalrechits_analyze_cfi import *
from Validation.GlobalHits.globalhits_analyze_cfi import *
from Validation.Configuration.globalValidation_cff import *

from HLTriggerOffline.Common.HLTValidation_cff import *


from SimGeneral.TrackingAnalysis.trackingParticles_cfi import *
from Validation.RecoMET.METRelValForDQM_cff import *
from Validation.TrackingMCTruth.trackingTruthValidation_cfi import *
from Validation.RecoTrack.TrackValidation_cff import *
from Validation.RecoMuon.muonValidation_cff import *
from Validation.MuonIsolation.MuIsoVal_cff import *
from Validation.MuonIdentification.muonIdVal_cff import *
from Validation.RecoMuon.muonValidationHLT_cff import *
from Validation.EventGenerator.BasicGenValidation_cff import *

validation = cms.Sequence(cms.SequencePlaceholder("mix")
                         +basicGenTest_seq
                         *globaldigisanalyze
                         *globalhitsanalyze
                         *globalrechitsanalyze
                         *globalValidation
                         *hltvalidation)

validation_preprod = cms.Sequence(
                          basicGenTest_seq
                          +trackingTruthValid
                          +tracksValidation
                          +METRelValSequence
                          +recoMuonValidation
                          +muIsoVal_seq
                          +muonIdValDQMSeq
                          +hltvalidation_preprod
                          )
validation.remove(condDataValidation)
validation_prod = cms.Sequence(
             basicGenTest_seq
            +hltvalidation_prod
            )

