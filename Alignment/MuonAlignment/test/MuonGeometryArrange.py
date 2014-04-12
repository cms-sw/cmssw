import FWCore.ParameterSet.Config as cms
process =cms.Process("TEST")
 
#Ideal geometry
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.MessageLogger = cms.Service("MessageLogger",
     info_txt = cms.untracked.PSet(
         threshold = cms.untracked.string('INFO')
     ),
     cerr = cms.untracked.PSet(
         threshold = cms.untracked.string('INFO')
     ),
     destinations = cms.untracked.vstring('info_txt','cerr')
 )
 
 
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
     input = cms.untracked.int32(1)
  )
 

# Full configuration for Muon Geometry Comparison Tool
process.MuonGeometryCompare = cms.EDFilter("MuonGeometryArrange",
    outputFile = cms.untracked.string('output.root'),

    detIdFlag = cms.untracked.bool(False),
    detIdFlagFile = cms.untracked.string('blah.txt'),
    weightById = cms.untracked.bool(False),
    levels = cms.untracked.vstring('Det'),
    weightBy = cms.untracked.string('SELF'),
    weightByIdFile = cms.untracked.string('blah2.txt'),
    treeName = cms.untracked.string('alignTree'),
# Root input files are not used yet.
    inputROOTFile1 = cms.untracked.string('IDEAL'),
    inputROOTFile2 = cms.untracked.string('idealmuon2.root'),
# Geometries are read from xml files
#     inputXMLCurrent = cms.untracked.string('B.xml'),
#     inputXMLCurrent = cms.untracked.string('A.xml'),
#     inputXMLCurrent = cms.untracked.string('moveRing.xml'),
#    inputXMLCurrent = cms.untracked.string('movedRing.xml'),
#    inputXMLCurrent = cms.untracked.string('fiddleMuon.xml'),
#    inputXMLCurrent = cms.untracked.string('fiddle2Muon.xml'),
    inputXMLCurrent = cms.untracked.string('fiddle3Muon.xml'),
    inputXMLReference = cms.untracked.string('idealMuon.xml'),
# A few defaults.  You pick.
    endcapNumber = cms.untracked.int32(2),
    stationNumber = cms.untracked.int32(3),
    ringNumber = cms.untracked.int32(2)
)

process.p = cms.Path( process.MuonGeometryCompare )
 



