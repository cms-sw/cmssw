import FWCore.ParameterSet.Config as cms

from Validation.GlobalDigis.globaldigis_analyze_cfi import *
from Validation.GlobalRecHits.globalrechits_analyze_cfi import *
from Validation.GlobalHits.globalhits_analyze_cfi import *
from DQMServices.Components.MEtoEDMConverter_cfi import *
DQMStore = cms.Service("DQMStore")

offlinedqm = cms.Sequence(globaldigisanalyze*globalrechitsanalyze*globalhitsanalyze*MEtoEDMConverter)

