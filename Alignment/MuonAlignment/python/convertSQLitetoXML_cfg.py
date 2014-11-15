import FWCore.ParameterSet.Config as cms

process = cms.Process("CONVERT")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    connect = cms.string("sqlite_file:NEW.db"),
    toGet = cms.VPSet(
        cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")),
        cms.PSet(record = cms.string("DTAlignmentErrorExtendedRcd"), tag = cms.string("DTAlignmentErrorExtendedRcd")),
        cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")),
        cms.PSet(record = cms.string("CSCAlignmentErrorExtendedRcd"), tag = cms.string("CSCAlignmentErrorExtendedRcd"))))

process.inertGlobalPositionRcd = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    connect = cms.string("sqlite_file:inertGlobalPositionRcd.db"),
    toGet = cms.VPSet(cms.PSet(record = cms.string("GlobalPositionRcd"), tag = cms.string("inertGlobalPositionRcd"))))

process.MuonGeometryDBConverter = cms.EDAnalyzer("MuonGeometryDBConverter",
    input = cms.string("db"),
    dtLabel = cms.string(""),
    cscLabel = cms.string(""),
    shiftErr = cms.double(1000.),
    angleErr = cms.double(6.28),
    getAPEs = cms.bool(True),
    output = cms.string("xml"),
    outputXML = cms.PSet(
        fileName = cms.string("REPLACEME.xml"),
        relativeto = cms.string("ideal"),
        survey = cms.bool(False),
        rawIds = cms.bool(False),
        eulerAngles = cms.bool(False),
        precision = cms.int32(10),
        suppressDTBarrel = cms.untracked.bool(True),
        suppressDTWheels = cms.untracked.bool(True),
        suppressDTStations = cms.untracked.bool(True),
        suppressDTChambers = cms.untracked.bool(False),
        suppressDTSuperLayers = cms.untracked.bool(False),
        suppressDTLayers = cms.untracked.bool(False),
        suppressCSCEndcaps = cms.untracked.bool(True),
        suppressCSCStations = cms.untracked.bool(True),
        suppressCSCRings = cms.untracked.bool(True),
        suppressCSCChambers = cms.untracked.bool(False),
        suppressCSCLayers = cms.untracked.bool(False)))

process.Path = cms.Path(process.MuonGeometryDBConverter)
