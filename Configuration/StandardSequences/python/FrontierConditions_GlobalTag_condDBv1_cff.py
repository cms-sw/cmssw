import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv1_cfi import *

# the following are needed for non PoolDBESSources
from CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi import *
from CalibCalorimetry.HcalPlugins.Hcal_Conditions_forGlobalTag_cff import *
from CalibTracker.Configuration.Tracker_DependentRecords_forGlobalTag_nofakes_cff import *
# FIXME: should be moved to a cfi in a castor package
CastorDbProducer = cms.ESProducer("CastorDbProducer")
