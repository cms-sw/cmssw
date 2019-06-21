import FWCore.ParameterSet.Config as cms

### Needed to access DTConfigManagerRcd and by DTTrig
#from L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfig_cff import *

DTTPG = cms.EDProducer( "DTTPG",
)
