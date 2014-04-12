import FWCore.ParameterSet.Config as cms

from DPGAnalysis.SiStripTools.filtertest.MessageLogger_cff import *

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(),
                            skipBadFiles = cms.untracked.bool(True)
                            )
from DPGAnalysis.SiStripTools.filtertest.raw_102169_debug_cff import fileNames
source.fileNames = fileNames

#---------------------------------------------------------------------
# Raw to Digi: TO BE TESTED !!
#---------------------------------------------------------------------

from DPGAnalysis.SiStripTools.filtertest.rawtodigi_cff import *

#-------------------------------------------------
# Global Tag
#-------------------------------------------------
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
#GlobalTag.globaltag = "CRAFT09_R_V5::All"
GlobalTag.globaltag = "GR09_R_35_V3A::All"

#-------------------------------------------------------------------------
from CalibTracker.SiStripESProducers.fake.SiStripLatencyFakeESSource_cfi import *
from CalibTracker.SiStripESProducers.services.SiStripLatencyGeneratorService_cfi import *
SiStripLatencyGenerator.latency = cms.uint32(140)
SiStripLatencyGenerator.mode = cms.uint32(47)
es_prefer_latency = cms.ESPrefer("SiStripLatencyFakeESSource","siStripLatencyFakeESSource")
#-------------------------------------------------------------------------

from DPGAnalysis.SiStripTools.eventwithhistoryproducer_cfi import *

#------------------------------------------------------------------------
# APV Cycle Phase Producer and monitor
#------------------------------------------------------------------------
from DPGAnalysis.SiStripTools.configurableapvcyclephaseproducer_GR09_cfi import *

#------------------------------------------------------------------------

sinit = cms.Sequence(scalersRawToDigi + consecutiveHEs + APVPhases )

from DPGAnalysis.SiStripTools.eventtimedistribution_cfi import *



