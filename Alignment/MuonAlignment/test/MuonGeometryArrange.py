import FWCore.ParameterSet.Config as cms
process =cms.Process("TEST")
 
#Ideal geometry
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load('Configuration.Geometry.GeometryExtended2021_cff')
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase1_2021_design")

import Geometry.DTGeometryBuilder.dtGeometryDB_cfi
import Geometry.CSCGeometryBuilder.cscGeometryDB_cfi
import Geometry.GEMGeometryBuilder.gemGeometryDB_cfi

process.DTGeometryIdeal = Geometry.DTGeometryBuilder.dtGeometryDB_cfi.DTGeometryESModule.clone()
process.DTGeometryIdeal.appendToDataLabel = 'MuonGeometryArrangeGeomIdeal'
process.DTGeometryIdeal.applyAlignment = cms.bool(False)
process.CSCGeometryIdeal = Geometry.CSCGeometryBuilder.cscGeometryDB_cfi.CSCGeometryESModule.clone()
process.CSCGeometryIdeal.appendToDataLabel = 'MuonGeometryArrangeGeomIdeal'
process.CSCGeometryIdeal.applyAlignment = cms.bool(False)
process.GEMGeometryIdeal = Geometry.GEMGeometryBuilder.gemGeometryDB_cfi.GEMGeometryESModule.clone()
process.GEMGeometryIdeal.appendToDataLabel = 'MuonGeometryArrangeGeomIdeal'
process.GEMGeometryIdeal.applyAlignment = cms.bool(False)

process.DTGeometryMuonGeometryArrange1 = Geometry.DTGeometryBuilder.dtGeometryDB_cfi.DTGeometryESModule.clone()
process.DTGeometryMuonGeometryArrange1.appendToDataLabel = 'MuonGeometryArrangeLabel1'
process.DTGeometryMuonGeometryArrange1.applyAlignment = cms.bool(False)
process.CSCGeometryMuonGeometryArrange1 = Geometry.CSCGeometryBuilder.cscGeometryDB_cfi.CSCGeometryESModule.clone()
process.CSCGeometryMuonGeometryArrange1.appendToDataLabel = 'MuonGeometryArrangeLabel1'
process.CSCGeometryMuonGeometryArrange1.applyAlignment = cms.bool(False)
process.GEMGeometryMuonGeometryArrange1 = Geometry.GEMGeometryBuilder.gemGeometryDB_cfi.GEMGeometryESModule.clone()
process.GEMGeometryMuonGeometryArrange1.appendToDataLabel = 'MuonGeometryArrangeLabel1'
process.GEMGeometryMuonGeometryArrange1.applyAlignment = cms.bool(False)

process.DTGeometryMuonGeometryArrange2 = Geometry.DTGeometryBuilder.dtGeometryDB_cfi.DTGeometryESModule.clone()
process.DTGeometryMuonGeometryArrange2.appendToDataLabel = 'MuonGeometryArrangeLabel2'
process.DTGeometryMuonGeometryArrange2.applyAlignment = cms.bool(False)
process.CSCGeometryMuonGeometryArrange2 = Geometry.CSCGeometryBuilder.cscGeometryDB_cfi.CSCGeometryESModule.clone()
process.CSCGeometryMuonGeometryArrange2.appendToDataLabel = 'MuonGeometryArrangeLabel2'
process.CSCGeometryMuonGeometryArrange2.applyAlignment = cms.bool(False)
process.GEMGeometryMuonGeometryArrange2 = Geometry.GEMGeometryBuilder.gemGeometryDB_cfi.GEMGeometryESModule.clone()
process.GEMGeometryMuonGeometryArrange2.appendToDataLabel = 'MuonGeometryArrangeLabel2'
process.GEMGeometryMuonGeometryArrange2.applyAlignment = cms.bool(False)

process.DTGeometryMuonGeometryArrange2a = Geometry.DTGeometryBuilder.dtGeometryDB_cfi.DTGeometryESModule.clone()
process.DTGeometryMuonGeometryArrange2a.appendToDataLabel = 'MuonGeometryArrangeLabel2a'
process.DTGeometryMuonGeometryArrange2a.applyAlignment = cms.bool(False)
process.CSCGeometryMuonGeometryArrange2a = Geometry.CSCGeometryBuilder.cscGeometryDB_cfi.CSCGeometryESModule.clone()
process.CSCGeometryMuonGeometryArrange2a.appendToDataLabel = 'MuonGeometryArrangeLabel2a'
process.CSCGeometryMuonGeometryArrange2a.applyAlignment = cms.bool(False)
process.GEMGeometryMuonGeometryArrange2a = Geometry.GEMGeometryBuilder.gemGeometryDB_cfi.GEMGeometryESModule.clone()
process.GEMGeometryMuonGeometryArrange2a.appendToDataLabel = 'MuonGeometryArrangeLabel2a'
process.GEMGeometryMuonGeometryArrange2a.applyAlignment = cms.bool(False)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    files = cms.untracked.PSet(
        info_txt = cms.untracked.PSet(
            threshold = cms.untracked.string('INFO')
        )
    )
)
 
 
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
     input = cms.untracked.int32(1)
  )
 

# Full configuration for Muon Geometry Comparison Tool
#process.MuonGeometryCompare = cms.EDFilter("MuonGeometryArrange",
process.MuonGeometryCompare = cms.EDAnalyzer("MuonGeometryArrange",
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

