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

_hfnose_hgcalTriggerPrimitivesTask = hgcalTriggerPrimitivesTask.copy()
_hfnose_hgcalTriggerPrimitivesTask.add(hfnoseVFE, hgcalConcentratorHFNose, hgcalBackEndLayer1HFNose, hgcalBackEndLayer2HFNose, hgcalTowerMapHFNose, hgcalTowerHFNose)

from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose
phase2_hfnose.toReplaceWith(
        hgcalTriggerPrimitivesTask, _hfnose_hgcalTriggerPrimitivesTask )

from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
from Configuration.Eras.Modifier_phase2_hgcalV11_cff import phase2_hgcalV11
from L1Trigger.L1THGCal.customTriggerGeometry import custom_geometry_V10, custom_geometry_V11_Imp3
from L1Trigger.L1THGCal.customCalibration import  custom_cluster_calibration_global
modifyHgcalTriggerPrimitivesWithV10Geometry_ = (phase2_hgcalV10 & ~phase2_hgcalV11).makeProcessModifier(custom_geometry_V10)
modifyHgcalTriggerPrimitivesWithV11Geometry_ = phase2_hgcalV11.makeProcessModifier(custom_geometry_V11_Imp3)

from Configuration.ProcessModifiers.convertHGCalDigisSim_cff import convertHGCalDigisSim
# can't declare a producer version of simHGCalUnsuppressedDigis in the normal flow of things,
# because it's already an EDAlias elsewhere
def _fakeHGCalDigiAlias(process):
	from EventFilter.HGCalRawToDigi.HGCDigiConverter_cfi import HGCDigiConverter as _HGCDigiConverter
	process.simHGCalUnsuppressedDigis = _HGCDigiConverter.clone()
	process.hgcalTriggerPrimitivesTask.add(process.simHGCalUnsuppressedDigis)
doFakeHGCalDigiAlias = convertHGCalDigisSim.makeProcessModifier(_fakeHGCalDigiAlias)
