import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.hgcalVFE_cff import *
from L1Trigger.L1THGCal.hgcalConcentrator_cff import *
from L1Trigger.L1THGCal.hgcalBackEndLayer1_cff import *
from L1Trigger.L1THGCal.hgcalBackEndLayer2_cff import *
from L1Trigger.L1THGCal.hgcalTowerMap_cff import *
from L1Trigger.L1THGCal.hgcalTower_cff import *


hgcalTriggerPrimitivesTask = cms.Task(hgcalVFE, hgcalConcentrator, hgcalBackEndLayer1, hgcalBackEndLayer2, hgcalTowerMap, hgcalTower)
hgcalTriggerPrimitives = cms.Sequence(hgcalTriggerPrimitivesTask)

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9
from L1Trigger.L1THGCal.customTriggerGeometry import custom_geometry_V9
from L1Trigger.L1THGCal.customCalibration import  custom_cluster_calibration_global
modifyHgcalTriggerPrimitivesWithV9Geometry_ = phase2_hgcalV9.makeProcessModifier(custom_geometry_V9)

from Configuration.ProcessModifiers.convertHGCalDigisSim_cff import convertHGCalDigisSim
# can't declare a producer version of simHGCalUnsuppressedDigis in the normal flow of things,
# because it's already an EDAlias elsewhere
def _fakeHGCalDigiAlias(process):
	from EventFilter.HGCalRawToDigi.HGCDigiConverter_cfi import HGCDigiConverter as _HGCDigiConverter
	process.simHGCalUnsuppressedDigis = _HGCDigiConverter.clone()
	process.hgcalTriggerPrimitivesTask.add(process.simHGCalUnsuppressedDigis)
doFakeHGCalDigiAlias = convertHGCalDigisSim.makeProcessModifier(_fakeHGCalDigiAlias)
