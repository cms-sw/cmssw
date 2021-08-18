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

process.DTGeometryAlInputMethod = cms.ESProducer("DTGeometryESModule",
    appendToDataLabel = cms.string('idealForInputMethod'),
    applyAlignment = cms.bool(False),
    alignmentsLabel = cms.string(''),
    fromDDD = cms.bool(True)
)

process.CSCGeometryAlInputMethod = cms.ESProducer("CSCGeometryESModule",
    appendToDataLabel = cms.string('idealForInputMethod'),
    debugV = cms.untracked.bool(False),
    useGangedStripsInME1a = cms.bool(False),
    alignmentsLabel = cms.string(''),
    useOnlyWiresInME1a = cms.bool(False),
    useRealWireGeometry = cms.bool(True),
    useCentreTIOffsets = cms.bool(False),
    applyAlignment = cms.bool(False),
    fromDDD = cms.bool(True),
    fromDD4hep = cms.bool(False)
)

process.GEMGeometryAlInputMethod = cms.ESProducer("GEMGeometryESModule",
    appendToDataLabel = cms.string('idealForInputMethod'),
    fromDDD = cms.bool(True),
    fromDD4Hep = cms.bool(False),
    alignmentsLabel = cms.string(''),
    applyAlignment = cms.bool(False)
)

process.DTGeometryAlInputDB = cms.ESProducer("DTGeometryESModule",
    appendToDataLabel = cms.string('idealForInputDB'),
    applyAlignment = cms.bool(False),
    alignmentsLabel = cms.string(''),
    fromDDD = cms.bool(True)
)

process.CSCGeometryAlInputDB = cms.ESProducer("CSCGeometryESModule",
    appendToDataLabel = cms.string('idealForInputDB'),
    debugV = cms.untracked.bool(False),
    useGangedStripsInME1a = cms.bool(False),
    alignmentsLabel = cms.string(''),
    useOnlyWiresInME1a = cms.bool(False),
    useRealWireGeometry = cms.bool(True),
    useCentreTIOffsets = cms.bool(False),
    applyAlignment = cms.bool(False),
    fromDDD = cms.bool(True),
    fromDD4hep = cms.bool(False)
)

process.GEMGeometryAlInputDB = cms.ESProducer("GEMGeometryESModule",
    appendToDataLabel = cms.string('idealForInputDB'),
    fromDDD = cms.bool(True),
    fromDD4Hep = cms.bool(False),
    alignmentsLabel = cms.string(''),
    applyAlignment = cms.bool(False)
)

process.DTGeometryAlOutputXML = cms.ESProducer("DTGeometryESModule",
    appendToDataLabel = cms.string('idealForOutputXML'),
    applyAlignment = cms.bool(False),
    alignmentsLabel = cms.string(''),
    fromDDD = cms.bool(True)
)

process.CSCGeometryAlOutputXML = cms.ESProducer("CSCGeometryESModule",
    appendToDataLabel = cms.string('idealForOutputXML'),
    debugV = cms.untracked.bool(False),
    useGangedStripsInME1a = cms.bool(False),
    alignmentsLabel = cms.string(''),
    useOnlyWiresInME1a = cms.bool(False),
    useRealWireGeometry = cms.bool(True),
    useCentreTIOffsets = cms.bool(False),
    applyAlignment = cms.bool(False),
    fromDDD = cms.bool(True),
    fromDD4hep = cms.bool(False)
)

process.GEMGeometryAlOutputXML = cms.ESProducer("GEMGeometryESModule",
    appendToDataLabel = cms.string('idealForOutputXML'),
    fromDDD = cms.bool(True),
    fromDD4Hep = cms.bool(False),
    alignmentsLabel = cms.string(''),
    applyAlignment = cms.bool(False)
)

process.DTGeometryAlInputXML = cms.ESProducer("DTGeometryESModule",
    appendToDataLabel = cms.string('idealForInputXML'),
    applyAlignment = cms.bool(False),
    alignmentsLabel = cms.string(''),
    fromDDD = cms.bool(True)
)

process.CSCGeometryAlInputXML = cms.ESProducer("CSCGeometryESModule",
    appendToDataLabel = cms.string('idealForInputXML'),
    debugV = cms.untracked.bool(False),
    useGangedStripsInME1a = cms.bool(False),
    alignmentsLabel = cms.string(''),
    useOnlyWiresInME1a = cms.bool(False),
    useRealWireGeometry = cms.bool(True),
    useCentreTIOffsets = cms.bool(False),
    applyAlignment = cms.bool(False),
    fromDDD = cms.bool(True),
    fromDD4hep = cms.bool(False)
)

process.GEMGeometryAlInputXML = cms.ESProducer("GEMGeometryESModule",
    appendToDataLabel = cms.string('idealForInputXML'),
    fromDDD = cms.bool(True),
    fromDD4Hep = cms.bool(False),
    alignmentsLabel = cms.string(''),
    applyAlignment = cms.bool(False)
)
################################################################################
# parameters to configure:
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase1_2017_design")
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
