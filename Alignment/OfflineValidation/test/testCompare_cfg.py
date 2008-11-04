import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_cff")

process.load("Alignment.OfflineValidation.TrackerGeometryCompare_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo', 
        'cout')
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)
process.p = cms.Path(process.TrackerGeometryCompare)
process.TrackerGeometryCompare.inputROOTFile1 = 'IDEAL'
process.TrackerGeometryCompare.inputROOTFile2 = 'testInputGeometry.root'
process.TrackerGeometryCompare.outputFile = 'outputComparison.root'
#process.TrackerGeometryCompare.levels = ['TOBBarrel','TIBBarrel','TIDEndcap','TECEndcap','TPBBarrel','TPEEndcap','DetUnit']
process.TrackerGeometryCompare.levels = ['DetUnit']


