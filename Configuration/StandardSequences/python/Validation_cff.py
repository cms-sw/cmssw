import FWCore.ParameterSet.Config as cms

from Validation.GlobalDigis.globaldigis_analyze_cfi import *
from Validation.GlobalRecHits.globalrechits_analyze_cfi import *
from Validation.GlobalHits.globalhits_analyze_cfi import *
from Validation.Configuration.globalValidation_cff import *

from HLTriggerOffline.Common.HLTValidation_cff import *

validation = cms.Sequence(globaldigisanalyze*globalhitsanalyze*hltvalidation)
