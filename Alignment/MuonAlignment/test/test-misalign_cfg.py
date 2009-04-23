import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
# -- Load default module/services configurations -- //
# Message logger service
process.load("FWCore.MessageService.MessageLogger_cfi")

# Ideal DT & CSC geometry 
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

# Database output service
process.load("CondCore.DBCommon.CondDBSetup_cfi")

# Misalignment example scenario producer
process.load("Alignment.MuonAlignment.MisalignedMuon_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBSetup,
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTAlignmentRcd'),
        tag = cms.string('DT100InversepbScenario')
    ), 
        cms.PSet(
            record = cms.string('DTAlignmentErrorRcd'),
            tag = cms.string('DT100InversepbScenarioErrors')
        ), 
        cms.PSet(
            record = cms.string('CSCAlignmentRcd'),
            tag = cms.string('CSC100InversepbScenario')
        ), 
        cms.PSet(
            record = cms.string('CSCAlignmentErrorRcd'),
            tag = cms.string('CSC100InversepbScenarioErrors')
        )),
    connect = cms.string('sqlite_file:Alignments.db')
)

process.prod = cms.EDFilter("TestMisalign",
    fileName = cms.untracked.string('misaligment.root')
)

process.p1 = cms.Path(process.prod)
process.MessageLogger.cout = cms.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    )
)
process.MisalignedMuon.saveToDbase = True


