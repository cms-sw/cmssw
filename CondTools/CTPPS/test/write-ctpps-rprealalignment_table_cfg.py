from __future__ import print_function
import FWCore.ParameterSet.Config as cms

process = cms.Process('test')

import sys
if len(sys.argv) > 2:
    startrun = sys.argv[1]
    subdir = sys.argv[2]+"/"
else:
    print("not able to run")
    exit()


process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(int(startrun)),
    lastValue  = cms.uint64(int(startrun)),
    interval = cms.uint64(1)
)

# load the alignment xml file
process.load("CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi")
#process.ctppsRPAlignmentCorrectionsDataESSourceXML.XMLFile = cms.string("CondFormats/PPSObjects/xml/sample_alignment_corrections.xml")
process.ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = cms.vstring(
    "CondFormats/PPSObjects/xml/sample_alignment_corrections.xml"
    #"CondTools/CTPPS/test/RPixGeometryCorrections.xml",
    #"CondTools/CTPPS/test/"+subdir+"real_alignment_iov"+startrun+".xml"
    )
process.ctppsRPAlignmentCorrectionsDataESSourceXML.MeasuredFiles = cms.vstring("CondFormats/PPSObjects/xml/sample_alignment_corrections.xml")
process.ctppsRPAlignmentCorrectionsDataESSourceXML.MisalignedFiles = cms.vstring("CondFormats/PPSObjects/xml/sample_alignment_corrections.xml")


#Database output service
process.load("CondCore.CondDB.CondDB_cfi")
# output database (in this case local sqlite file)
process.CondDB.connect = 'sqlite_file:CTPPSRPRealAlignment_table.db'


process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
    cms.PSet(
        record = cms.string('RPRealAlignmentRecord'),
        tag = cms.string('CTPPSRPAlignment_real_table'),
    )
  )
)


# print the mapping and analysis mask
process.writeCTPPSRPAlignments = cms.EDAnalyzer("CTPPSRPAlignmentInfoAnalyzer",
    cms.PSet(
        iov = cms.uint64(int(startrun)),
        record = cms.string("RPRealAlignmentRecord")
    )
)

process.path = cms.Path(
  process.writeCTPPSRPAlignments
)
