import FWCore.ParameterSet.Config as cms

#--- Allow for multiple calls to the database ---#
from CondCore.DBCommon.CondDBSetup_cfi import *
#--- Geometry Setup ---#
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from HLTrigger.Configuration.rawToDigi.EcalGeometrySetup_cff import *
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_with_suppressed_cff import *
import copy
from EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi import *
ecalDigis = copy.deepcopy(ecalEBunpacker)
import copy
from EventFilter.ESRawToDigi.esRawToDigi_cfi import *
#--- Ecal Preshower ---#
ecalPreshowerDigis = copy.deepcopy(esRawToDigi)
import copy
from EventFilter.HcalRawToDigi.HcalRawToDigi_cfi import *
#--- Hcal ---#
hcalDigis = copy.deepcopy(hcalDigis)
CaloRawToDigi = cms.Sequence(ecalDigis+ecalPreshowerDigis+hcalDigis)
ecalDigis.DoRegional = False
ecalDigis.InputLabel = 'rawDataCollector'
ecalPreshowerDigis.Label = 'rawDataCollector'
hcalDigis.InputLabel = 'rawDataCollector'

