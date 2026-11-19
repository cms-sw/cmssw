import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

# -- Load default module/services configurations -- //
# Message logger service
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

# MTD Geometry and phase 2 configuration
process.load("Configuration.Geometry.GeometryExtendedRun4D125Reco_cff")

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

#Choose MTD misalignment scenario
process.MisalignedMTD = cms.EDAnalyzer("MTDMisalignedProducer",
                                        scenario = _MTDScenarios.MTDBTLStartup,
                                        #scenario = _MTDScenarios.MTDNoMovementsScenario,
                                        saveToDbase = cms.untracked.bool(True)
                                      )


#Load MTD Digi Geometry producer
process.MTDGeometryMisalignedProducer = cms.ESProducer("MTDDigiGeometryESModule",
    appendToDataLabel = cms.string('idealForMTDMisalignedProducer'),
    applyAlignment = cms.bool(False), 
    alignmentsLabel = cms.string(''),
    fromDDD = cms.bool(True)
)


# Database output service if you want to store soemthing in MTD
from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    CondDBSetup,
    toPut = cms.VPSet(
        cms.PSet(
        record = cms.string('MTDAlignmentRcd'),
        tag = cms.string('Alignments')
        ), 
        cms.PSet(
            record = cms.string('MTDAlignmentErrorExtendedRcd'),
            tag = cms.string('AlignmentErrorsExtended')
        ), 
        ),
    connect = cms.string('sqlite_file:Alignments.db')
)
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.threshold = "DEBUG"



process.p1 = cms.Path(process.MisalignedMTD)
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    )
)


