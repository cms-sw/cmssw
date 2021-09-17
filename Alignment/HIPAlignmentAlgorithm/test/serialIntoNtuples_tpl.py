import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_cff")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    ),
    files = cms.untracked.PSet(
        detailedInfo = cms.untracked.PSet(

        )
    )
)

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('Alignments')
    ), 
        cms.PSet(
            record = cms.string('TrackerAlignmentErrorExtendedRcd'),
            tag = cms.string('AlignmentErrorsExtended')
        )),
    connect = cms.string('sqlite_file:<PATH>/alignments_<N>.db')
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)
process.dump = cms.EDFilter("TrackerGeometryIntoNtuples",
    outputFile = cms.untracked.string('<PATH>/alignments_<N>.root'),
    outputTreename = cms.untracked.string('alignTree')
)

process.p = cms.Path(process.dump)


