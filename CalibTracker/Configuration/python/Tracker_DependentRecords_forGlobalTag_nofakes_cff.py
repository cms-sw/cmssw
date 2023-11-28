import FWCore.ParameterSet.Config as cms

#Gain
# first SiStripGainESProducer takes SiStripGainRcd from DB
from CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi import *

# SiStripLorentzAngleDep producer to select the LA value according to Tracker mode
from CalibTracker.SiStripESProducers.SiStripLorentzAngleDepESProducer_cfi import *

# SiStripBackPlaneCorrectionDep producer to select the LA value according to Tracker mode
from CalibTracker.SiStripESProducers.SiStripBackPlaneCorrectionDepESProducer_cfi import *

#LorentzAngle
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *

#cabling
sistripconn = cms.ESProducer("SiStripConnectivity")

from CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi import *
siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
     cms.PSet( record = cms.string("SiStripDetVOffRcd"),    tag    = cms.string("") ),
     cms.PSet( record = cms.string("SiStripDetCablingRcd"), tag    = cms.string("") ),
     cms.PSet( record = cms.string("RunInfoRcd"),           tag    = cms.string("") ),
     cms.PSet( record = cms.string("SiStripBadChannelRcd"), tag    = cms.string("") ),
     cms.PSet( record = cms.string("SiStripBadFiberRcd"),   tag    = cms.string("") ),
     cms.PSet( record = cms.string("SiStripBadModuleRcd"),  tag    = cms.string("") ),
     cms.PSet( record = cms.string("SiStripBadStripRcd"),   tag    = cms.string("") )
     )
siStripQualityESProducer.ReduceGranularity = cms.bool(False)
# True means all the debug output from adding the RunInfo (default is False)
siStripQualityESProducer.PrintDebugOutput = cms.bool(False)
# "True" means that the RunInfo is used even if all the feds are off (including other subdetectors).
# This means that if the RunInfo was filled with a fake empty object we will still set the full tracker as bad.
# With "False", instead, in that case the RunInfo information is discarded.
# Default is "False".
siStripQualityESProducer.UseEmptyRunInfo = cms.bool(False)

from CalibTracker.SiPixelESProducers.siPixelQualityESProducer_cfi import *
from CalibTracker.SiPixelESProducers.siPixelQualityForRawToDigiESProducer_cfi import *

# Multiple scattering parametrisation
from RecoTracker.TkMSParametrization.multipleScatteringParametrisationMakerESProducer_cfi import *
