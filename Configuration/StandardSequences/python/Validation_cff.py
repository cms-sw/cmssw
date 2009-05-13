import FWCore.ParameterSet.Config as cms


RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
        mix = cms.PSet(initialSeed = cms.untracked.uint32(12345),
                       engineName = cms.untracked.string('HepJamesRandom')
        ),
        restoreStateLabel = cms.untracked.string("randomEngineStateProducer"),
)

mix.playback=True

from Validation.GlobalDigis.globaldigis_analyze_cfi import *
from Validation.GlobalRecHits.globalrechits_analyze_cfi import *
from Validation.GlobalHits.globalhits_analyze_cfi import *
from Validation.Configuration.globalValidation_cff import *

from HLTriggerOffline.Common.HLTValidation_cff import *

validation = cms.Sequence(cms.SequencePlaceHolder("mix")+globaldigisanalyze*globalhitsanalyze*globalrechitsanalyze*globalValidation*hltvalidation)
