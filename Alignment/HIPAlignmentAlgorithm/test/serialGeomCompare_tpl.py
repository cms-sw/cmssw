import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_cff")

process.load("Alignment.OfflineValidation.TrackerGeometryCompare_cfi")

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

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)
process.p = cms.Path(process.TrackerGeometryCompare)
#process.TrackerGeometryCompare.inputROOTFile1 = 'IDEAL'
process.TrackerGeometryCompare.inputROOTFile1 = '<PATH>/original.root'
process.TrackerGeometryCompare.inputROOTFile2 = '<PATH>/alignments_<N>.root'
process.TrackerGeometryCompare.outputFile = '<PATH>/comparisonV3_<N>.root'
#process.TrackerGeometryCompare.levels = ['TOBBarrel','TIBBarrel','TIDEndcap','TECEndcap','TPBBarrel','TPEEndcap','DetUnit']
process.TrackerGeometryCompare.levels = ['Tracker','DetUnit']


