import FWCore.ParameterSet.Config as cms

process = cms.Process("test")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Alignment.MuonAlignment.Scenarios_cff")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)
process.MuonGeometryDBConverter = cms.EDFilter("MuonGeometryDBConverter",
    outputXML = cms.PSet(
        relativeto = cms.string('ideal'),
        eulerAngles = cms.bool(False),
        survey = cms.bool(False),
        rawIds = cms.bool(False),
        fileName = cms.string('tmp.xml')
    ),
    MisalignmentScenario = cms.PSet(
        distribution = cms.string('fixed'),
        seed = cms.int32(1234567),
        setError = cms.bool(True),
        DTBarrels = cms.PSet(
            DTChambers = cms.PSet(
                dXlocal = cms.double(1.0)
            )
        ),
        CSCEndcaps = cms.PSet(
            CSCChambers = cms.PSet(
                dXlocal = cms.double(1.0)
            )
        )
    ),
    shiftErr = cms.double(5.0),
    output = cms.string('xml'),
    input = cms.string('scenario'),
    angleErr = cms.double(1.0)
)

process.p = cms.Path(process.MuonGeometryDBConverter)


