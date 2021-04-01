import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
# -- Load default module/services configurations -- //
# Message logger service
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load('Configuration.Geometry.GeometryExtended2021_cff')

# Misalignment example scenario producer
import Alignment.MuonAlignment.Scenarios_cff as _MuonScenarios
# Does not yet work like that:
#process.load("Alignment.MuonAlignment.MisalignedMuon_cfi")
#process.MisalignedMuon.saveToDbase = True # to store to DB
#process.MisalignedMuon.scenario = _MuonScenarios.Muon0inversePbScenario2008
process.MisalignedMuon = cms.EDAnalyzer("MuonMisalignedProducer",
                                        _MuonScenarios.ExampleScenario,
                                        saveToDbase = cms.untracked.bool(True)
                                        )

process.MisalignedMuon.scenario = _MuonScenarios.Muon100InversepbScenario
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.DTGeometryMisalignedMuonProducer = cms.ESProducer("DTGeometryESModule",
    appendToDataLabel = cms.string('idealForMuonMisalignedProducer'),
    applyAlignment = cms.bool(False), 
    alignmentsLabel = cms.string(''),
    fromDDD = cms.bool(True)
)

process.CSCGeometryMisalignedMuonProducer = cms.ESProducer("CSCGeometryESModule",
    appendToDataLabel = cms.string('idealForMuonMisalignedProducer'),
    debugV = cms.untracked.bool(False),
    useGangedStripsInME1a = cms.bool(False),
    alignmentsLabel = cms.string(''),
    useOnlyWiresInME1a = cms.bool(False),
    useRealWireGeometry = cms.bool(True),
    useCentreTIOffsets = cms.bool(False),
    applyAlignment = cms.bool(False), 
    fromDDD = cms.bool(True),
    fromDD4hep = cms.bool(False)
)

process.GEMGeometryMisalignedMuonProducer = cms.ESProducer("GEMGeometryESModule",
    appendToDataLabel = cms.string('idealForMuonMisalignedProducer'),
    fromDDD = cms.bool(True),
    fromDD4Hep = cms.bool(False),
    alignmentsLabel = cms.string(''),
    applyAlignment = cms.bool(False)
)


# Database output service if you want to store soemthing in MisalignedMuon
from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    CondDBSetup,
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTAlignmentRcd'),
        tag = cms.string('DT100InversepbScenario')
    ), 
        cms.PSet(
            record = cms.string('DTAlignmentErrorExtendedRcd'),
            tag = cms.string('DT100InversepbScenarioErrors')
        ), 
        cms.PSet(
            record = cms.string('CSCAlignmentRcd'),
            tag = cms.string('CSC100InversepbScenario')
        ), 
        cms.PSet(
            record = cms.string('CSCAlignmentErrorExtendedRcd'),
            tag = cms.string('CSC100InversepbScenarioErrors')
        ),
        cms.PSet(
            record = cms.string('GEMAlignmentRcd'),
            tag = cms.string('GEM')
        ), 
        cms.PSet(
            record = cms.string('GEMAlignmentErrorExtendedRcd'),
            tag = cms.string('test')
        )),

    connect = cms.string('sqlite_file:Alignments.db')
)

process.prod = cms.EDAnalyzer("TestMisalign",
    fileName = cms.untracked.string('misaligment.root')
)

#process.p1 = cms.Path(process.MisalignedMuon+process.prod)
process.p1 = cms.Path(process.MisalignedMuon)
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    )
)


