import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.hgcalVFEProducer_cfi import *
from L1Trigger.L1THGCal.hgcalConcentratorProducer_cfi import *
from L1Trigger.L1THGCal.hgcalBackEndLayer1Producer_cfi import *
from L1Trigger.L1THGCal.hgcalBackEndLayer2Producer_cfi import *
from L1Trigger.L1THGCal.hgcalTowerMapProducer_cfi import *
from L1Trigger.L1THGCal.hgcalTowerProducer_cfi import *


hgcalTriggerPrimitives = cms.Sequence(hgcalVFEProducer*hgcalConcentratorProducer*hgcalBackEndLayer1Producer*hgcalBackEndLayer2Producer*hgcalTowerMapProducer*hgcalTowerProducer)

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9
from L1Trigger.L1THGCal.customTriggerGeometry import custom_geometry_V9
modifyHgcalTriggerPrimitivesWithV9Geometry_ = phase2_hgcalV9.makeProcessModifier(custom_geometry_V9)

from Configuration.ProcessModifiers.convertHGCalDigisSim_cff import convertHGCalDigisSim
# can't declare a producer version of simHGCalUnsuppressedDigis in the normal flow of things,
# because it's already an EDAlias elsewhere
def _fakeHGCalDigiAlias(process):
	from EventFilter.HGCalRawToDigi.HGCDigiConverter_cfi import HGCDigiConverter as _HGCDigiConverter
	process.simHGCalUnsuppressedDigis = _HGCDigiConverter.clone()
	process.hgcalTriggerPrimitives.insert(0,process.simHGCalUnsuppressedDigis)
doFakeHGCalDigiAlias = convertHGCalDigisSim.makeProcessModifier(_fakeHGCalDigiAlias)
