from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing


################################################################################
# command line options
options = VarParsing.VarParsing()
options.register('input',
                 default = "ideal",
                 mytype = VarParsing.VarParsing.varType.string,
                 info = "input format")
options.register('inputFile',
                 default = None,
                 mytype = VarParsing.VarParsing.varType.string,
                 info = "input file name")
options.register('output',
                 default = "none",
                 mytype = VarParsing.VarParsing.varType.string,
                 info = "output format")
options.register('outputFile',
                 default = None,
                 mytype = VarParsing.VarParsing.varType.string,
                 info = "output file name")
options.parseArguments()

################################################################################
# setting up the process
process = cms.Process("CONVERT")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load('Configuration.Geometry.GeometryExtended2021_cff')
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Alignment.MuonAlignment.muonGeometryDBConverter_cfi")

import Geometry.DTGeometryBuilder.dtGeometryDB_cfi
import Geometry.CSCGeometryBuilder.cscGeometryDB_cfi
import Geometry.GEMGeometryBuilder.gemGeometryDB_cfi
process.DTGeometryIdealForInputMethod = Geometry.DTGeometryBuilder.dtGeometryDB_cfi.DTGeometryESModule.clone()
process.DTGeometryIdealForInputMethod.appendToDataLabel = 'idealForInputMethod'
process.DTGeometryIdealForInputMethod.applyAlignment = cms.bool(False)
process.CSCGeometryIdealForInputMethod = Geometry.CSCGeometryBuilder.cscGeometryDB_cfi.CSCGeometryESModule.clone()
process.CSCGeometryIdealForInputMethod.appendToDataLabel = 'idealForInputMethod'
process.CSCGeometryIdealForInputMethod.applyAlignment = cms.bool(False)
process.GEMGeometryIdealForInputMethod = Geometry.GEMGeometryBuilder.gemGeometryDB_cfi.GEMGeometryESModule.clone()
process.GEMGeometryIdealForInputMethod.appendToDataLabel = 'idealForInputMethod'
process.GEMGeometryIdealForInputMethod.applyAlignment = cms.bool(False)

process.DTGeometryIdealForInputDB = Geometry.DTGeometryBuilder.dtGeometryDB_cfi.DTGeometryESModule.clone()
process.DTGeometryIdealForInputDB.appendToDataLabel = 'idealForInputDB'
process.DTGeometryIdealForInputDB.applyAlignment = cms.bool(False)
process.CSCGeometryIdealForInputDB = Geometry.CSCGeometryBuilder.cscGeometryDB_cfi.CSCGeometryESModule.clone()
process.CSCGeometryIdealForInputDB.appendToDataLabel = 'idealForInputDB'
process.CSCGeometryIdealForInputDB.applyAlignment = cms.bool(False)
process.GEMGeometryIdealForInputDB = Geometry.GEMGeometryBuilder.gemGeometryDB_cfi.GEMGeometryESModule.clone()
process.GEMGeometryIdealForInputDB.appendToDataLabel = 'idealForInputDB'
process.GEMGeometryIdealForInputDB.applyAlignment = cms.bool(False)

process.DTGeometryIdealForInputXML = Geometry.DTGeometryBuilder.dtGeometryDB_cfi.DTGeometryESModule.clone()
process.DTGeometryIdealForInputXML.appendToDataLabel = 'idealForInputXML'
process.DTGeometryIdealForInputXML.applyAlignment = cms.bool(False)
process.CSCGeometryIdealForInputXML = Geometry.CSCGeometryBuilder.cscGeometryDB_cfi.CSCGeometryESModule.clone()
process.CSCGeometryIdealForInputXML.appendToDataLabel = 'idealForInputXML'
process.CSCGeometryIdealForInputXML.applyAlignment = cms.bool(False)
process.GEMGeometryIdealForInputXML = Geometry.GEMGeometryBuilder.gemGeometryDB_cfi.GEMGeometryESModule.clone()
process.GEMGeometryIdealForInputXML.appendToDataLabel = 'idealForInputXML'
process.GEMGeometryIdealForInputXML.applyAlignment = cms.bool(False)

process.DTGeometryIdealForInputSurveyDB = Geometry.DTGeometryBuilder.dtGeometryDB_cfi.DTGeometryESModule.clone()
process.DTGeometryIdealForInputSurveyDB.appendToDataLabel = 'idealForInputSurveyDB'
process.DTGeometryIdealForInputSurveyDB.applyAlignment = cms.bool(False)
process.CSCGeometryIdealForInputSurveyDB = Geometry.CSCGeometryBuilder.cscGeometryDB_cfi.CSCGeometryESModule.clone()
process.CSCGeometryIdealForInputSurveyDB.appendToDataLabel = 'idealForInputSurveyDB'
process.CSCGeometryIdealForInputSurveyDB.applyAlignment = cms.bool(False)
process.GEMGeometryIdealForInputSurveyDB = Geometry.GEMGeometryBuilder.gemGeometryDB_cfi.GEMGeometryESModule.clone()
process.GEMGeometryIdealForInputSurveyDB.appendToDataLabel = 'idealForInputSurveyDB'
process.GEMGeometryIdealForInputSurveyDB.applyAlignment = cms.bool(False)

################################################################################
# parameters to configure:
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase1_2021_design")
process.muonGeometryDBConverter.input = options.input
process.muonGeometryDBConverter.output = options.output

if options.input == "db":
    process.GlobalTag.toGet.extend(
        [cms.PSet(connect = cms.string("sqlite_file:"+options.inputFile),
                  record = cms.string("DTAlignmentRcd"),
                  tag = cms.string("DTAlignmentRcd")),
         cms.PSet(connect = cms.string("sqlite_file:"+options.inputFile),
                  record = cms.string("DTAlignmentErrorExtendedRcd"),
                  tag = cms.string("DTAlignmentErrorExtendedRcd")),
         cms.PSet(connect = cms.string("sqlite_file:"+options.inputFile),
                  record = cms.string("CSCAlignmentRcd"),
                  tag = cms.string("CSCAlignmentRcd")),
         cms.PSet(connect = cms.string("sqlite_file:"+options.inputFile),
                  record = cms.string("CSCAlignmentErrorExtendedRcd"),
                  tag = cms.string("CSCAlignmentErrorExtendedRcd")),
         cms.PSet(connect = cms.string("sqlite_file:"+options.inputFile),
                  record = cms.string("GEMAlignmentRcd"),
                  tag = cms.string("GEMAlignmentRcd")),
         cms.PSet(connect = cms.string("sqlite_file:"+options.inputFile),
                  record = cms.string("GEMAlignmentErrorExtendedRcd"),
                  tag = cms.string("GEMAlignmentErrorExtendedRcd"))
        ])
elif options.input == "xml":
    process.muonGeometryDBConverter.fileName = options.inputFile

if options.output == "db":
    from CondCore.CondDB.CondDB_cfi import CondDB
    process.PoolDBOutputService = cms.Service(
        "PoolDBOutputService",
        CondDB,
        toPut = cms.VPSet(
            cms.PSet(record = cms.string("DTAlignmentRcd"),
                     tag = cms.string("DTAlignmentRcd")),
            cms.PSet(record = cms.string("DTAlignmentErrorExtendedRcd"),
                     tag = cms.string("DTAlignmentErrorExtendedRcd")),
            cms.PSet(record = cms.string("CSCAlignmentRcd"),
                     tag = cms.string("CSCAlignmentRcd")),
            cms.PSet(record = cms.string("CSCAlignmentErrorExtendedRcd"),
                     tag = cms.string("CSCAlignmentErrorExtendedRcd")),
            cms.PSet(record = cms.string("GEMAlignmentRcd"),
                     tag = cms.string("GEMAlignmentRcd")),
            cms.PSet(record = cms.string("GEMAlignmentErrorExtendedRcd"),
                     tag = cms.string("GEMAlignmentErrorExtendedRcd"))
        )
    )
    process.PoolDBOutputService.connect = "sqlite_file:"+options.outputFile
elif options.output == "xml":
    process.muonGeometryDBConverter.outputXML.fileName = options.outputFile
    process.muonGeometryDBConverter.outputXML.suppressDTSuperLayers = True
    process.muonGeometryDBConverter.outputXML.suppressDTLayers = True
    process.muonGeometryDBConverter.outputXML.suppressCSCLayers = True


################################################################################

usedGlobalTag = process.GlobalTag.globaltag.value()
print("Using Global Tag:", usedGlobalTag)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.source = cms.Source("EmptySource")
process.p = cms.Path(process.muonGeometryDBConverter)
