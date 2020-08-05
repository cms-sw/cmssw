import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
# Message logger service
process.load("FWCore.MessageService.MessageLogger_cfi")

# DT and CSC geometry 
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

# Reading misalignments from DB
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")


process.DTGeometryMisalignedMuonProducer = cms.ESProducer("DTGeometryESModule",
    appendToDataLabel = cms.string('idealForTestReader'),
    applyAlignment = cms.bool(False),
    alignmentsLabel = cms.string(''),
    fromDDD = cms.bool(True)
)

process.CSCGeometryMisalignedMuonProducer = cms.ESProducer("CSCGeometryESModule",
    appendToDataLabel = cms.string('idealForTestReader'),
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


from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
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
        )),
    connect = cms.string('sqlite_file:Alignments.db')
)

process.prod = cms.EDAnalyzer("TestMuonReader")

process.p1 = cms.Path(process.prod)


