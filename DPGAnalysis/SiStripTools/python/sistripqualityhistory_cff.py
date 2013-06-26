import FWCore.ParameterSet.Config as cms

import CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi

ssqcabling = CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi.siStripQualityESProducer.clone()
ssqcabling.appendToDataLabel = cms.string("onlyCabling")
ssqcabling.ListOfRecordToMerge=cms.VPSet(
 cms.PSet(record=cms.string('SiStripDetCablingRcd'),tag=cms.string(''))
)

ssqruninfo = CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi.siStripQualityESProducer.clone()
ssqruninfo.appendToDataLabel = cms.string("CablingRunInfo")
ssqruninfo.ListOfRecordToMerge=cms.VPSet(
 cms.PSet(record=cms.string('SiStripDetCablingRcd'),tag=cms.string('')),
 cms.PSet(record=cms.string('RunInfoRcd'),tag=cms.string(''))
)

ssqbadch = CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi.siStripQualityESProducer.clone()
ssqbadch.appendToDataLabel = cms.string("BadChannel")
ssqbadch.ListOfRecordToMerge=cms.VPSet(
 cms.PSet(record=cms.string('SiStripDetCablingRcd'),tag=cms.string('')),
 cms.PSet(record=cms.string('RunInfoRcd'),tag=cms.string('')),
 cms.PSet(record=cms.string('SiStripBadChannelRcd'),tag=cms.string(''))
) 

ssqdcs = CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi.siStripQualityESProducer.clone()
ssqdcs.appendToDataLabel = cms.string("dcsBadModules")
ssqdcs.ListOfRecordToMerge=cms.VPSet(
 cms.PSet(record=cms.string('SiStripDetCablingRcd'),tag=cms.string('')),
 cms.PSet(record=cms.string('RunInfoRcd'),tag=cms.string('')),
 cms.PSet(record=cms.string('SiStripBadChannelRcd'),tag=cms.string('')),
 cms.PSet(record=cms.string('SiStripDetVOffRcd'),tag=cms.string(''))
)

ssqbadfiber = CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi.siStripQualityESProducer.clone()
ssqbadfiber.appendToDataLabel = cms.string("BadFiber")
ssqbadfiber.ListOfRecordToMerge=cms.VPSet(
 cms.PSet(record=cms.string('SiStripDetCablingRcd'),tag=cms.string('')),
 cms.PSet(record=cms.string('RunInfoRcd'),tag=cms.string('')),
 cms.PSet(record=cms.string('SiStripBadChannelRcd'),tag=cms.string('')),
 cms.PSet(record=cms.string('SiStripDetVOffRcd'),tag=cms.string('')),
 cms.PSet(record=cms.string('SiStripBadFiberRcd'),tag=cms.string(''))
)



from DPGAnalysis.SiStripTools.sistripqualityhistory_cfi import *

ssqhistory.monitoredSiStripQuality = cms.VPSet(
    cms.PSet( name = cms.string("Cabling"), ssqLabel = cms.string("onlyCabling")),
    cms.PSet( name = cms.string("RunInfo"), ssqLabel = cms.string("CablingRunInfo")),
    cms.PSet( name = cms.string("BadChannel"), ssqLabel = cms.string("BadChannel")),
    cms.PSet( name = cms.string("DCS"), ssqLabel = cms.string("dcsBadModules")),
    cms.PSet( name = cms.string("BadFiber"), ssqLabel = cms.string("BadFiber"))
   )
