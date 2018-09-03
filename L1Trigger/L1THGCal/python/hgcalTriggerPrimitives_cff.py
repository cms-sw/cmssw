import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.hgcalTriggerPrimitiveDigiProducer_cfi import *


hgcalTriggerPrimitives = cms.Sequence(hgcalTriggerPrimitiveDigiProducer)

hgcalTriggerPrimitives_reproduce = cms.Sequence(hgcalTriggerPrimitiveDigiFEReproducer)


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
