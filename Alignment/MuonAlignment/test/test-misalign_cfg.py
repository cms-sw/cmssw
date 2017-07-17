import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
# -- Load default module/services configurations -- //
# Message logger service
process.load("FWCore.MessageService.MessageLogger_cfi")

# Ideal DT & CSC geometry 
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

# Misalignment example scenario producer
import Alignment.MuonAlignment.Scenarios_cff as _MuonScenarios
# Does not yet work like that:
#process.load("Alignment.MuonAlignment.MisalignedMuon_cfi")
#process.MisalignedMuon.saveToDbase = True # to store to DB
#process.MisalignedMuon.scenario = _MuonScenarios.Muon0inversePbScenario2008
process.MisalignedMuon = cms.ESProducer("MisalignedMuonESProducer",
                                        _MuonScenarios.ExampleScenario,
                                        saveToDbase = cms.untracked.bool(True)
                                        )

# or standard stuff 
# Reco geometry producer
#process.load("Geometry.DTGeometry.dtGeometry_cfi")
#process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

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
        )),
    connect = cms.string('sqlite_file:Alignments.db')
)

process.prod = cms.EDAnalyzer("TestMisalign",
    fileName = cms.untracked.string('misaligment.root')
)

process.p1 = cms.Path(process.prod)
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    )
)


