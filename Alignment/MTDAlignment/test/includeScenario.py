import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

# -- Load default module/services configurations -- //
# Message logger service
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

# MTD Geometry and phase 2 configuration
process.load("Configuration.Geometry.GeometryExtendedRun4DefaultReco_cff")

import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

# Misalignment example scenario producer
import Alignment.MTDAlignment.Scenarios_cff as _MTDScenarios

#Empty source for IOV
process.source = cms.Source("EmptyIOVSource",
                             lastValue = cms.uint64(1),
                             timetype = cms.string('runnumber'),
                             firstValue = cms.uint64(1),
                             interval = cms.uint64(1)
                             )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#Load MTD Digi Geometry producer
process.MTDGeometryMisalignedProducer = cms.ESProducer("MTDDigiGeometryESModule",
    appendToDataLabel = cms.string('idealForMTDMisalignedProducer'),
    applyAlignment = cms.bool(False), 
    alignmentsLabel = cms.string(''),
    fromDDD = cms.bool(True)
)

process.MTDAlignment = cms.ESSource("PoolDBESSource",
                                     CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
                                     connect = cms.string("sqlite_file:Alignments.db"),
                                     toGet = cms.VPSet(
                                         cms.PSet(record = cms.string("MTDAlignmentRcd"),
                                         tag = cms.string("MTDAlignmentRcd")),
                                         cms.PSet(record = cms.string("MTDAlignmentErrorExtendedRcd"),
                                         tag = cms.string("MTDAlignmentErrorExtendedRcd"))))




process.p1 = cms.Path(process.MisalignedMTD)
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    )
)


