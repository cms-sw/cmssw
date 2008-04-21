# The following comments couldn't be translated into the new config version:

#sequence validation={globaldigisanalyze, globalrechitsanalyze, globalhitsanalyze, MEtoEDMConverter}

import FWCore.ParameterSet.Config as cms

from Validation.GlobalDigis.globaldigis_analyze_cfi import *
from Validation.GlobalRecHits.globalrechits_analyze_cfi import *
from Validation.GlobalHits.globalhits_analyze_cfi import *
from DQMServices.Components.MEtoEDMConverter_cfi import *
validation = cms.Sequence(globaldigisanalyze*globalhitsanalyze*MEtoEDMConverter)

