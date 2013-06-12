import os
import FWCore.ParameterSet.Config as cms

#*******************************************************************************
# Three environment variables must be set as in the following example:          
#    export ALIGNMENT_GLOBALTAG=FT_53_V6_AN2                                    
#    export ALIGNMENT_FIRSTRUN=157866                                           
#    export ALIGNMENT_OUTPUTFILE=muonGeometry-FT_53_V6_AN2-200000.db            
#*******************************************************************************

# Download Muon Alignment Record from the Global Tag with the following name:
#   See list of all Global Tags:
#   https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideFrontierConditions
globalTag  = os.getenv("ALIGNMENT_GLOBALTAG")

# Download Muon Alignment Record from Interval Of Validity that starts with the
# following run number:
firstRun   = os.getenv("ALIGNMENT_FIRSTRUN")

# Download Muon Alignment Record to SQLite DB file with the following name:
#   Suggested name convention: muonGeometry-[Global Tag name]-[First Run number]
outputFile = os.getenv("ALIGNMENT_OUTPUTFILE")

globalTag = globalTag + "::All" # Adds required suffix

print "Start download Muon Alignment Record from"
print "  globalTag  =    ", globalTag
print "  firstRun   =    ", firstRun
print "  to outputFile = ", outputFile

process = cms.Process("DOWNLOAD")

process.source = cms.Source("EmptySource",
  numberEventsInRun = cms.untracked.uint32(1),
  firstRun = cms.untracked.uint32(int(firstRun))
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.MessageLogger = cms.Service("MessageLogger",
  destinations = cms.untracked.vstring("cout"),
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string("ERROR")
  )
)

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag.globaltag = globalTag

process.inertGlobalPositionRcd = cms.ESSource("PoolDBESSource",
  process.CondDBSetup,
  connect = cms.string("sqlite_file:inertGlobalPositionRcd.db"),
  toGet = cms.VPSet(
    cms.PSet(
      record = cms.string("GlobalPositionRcd"),
      tag = cms.string("inertGlobalPositionRcd")
    )
  )
)

process.inertGlobalPositionRcd_prefer = cms.ESPrefer("PoolDBESSource","inertGlobalPositionRcd")

process.MuonGeometryDBConverter = cms.EDAnalyzer("MuonGeometryDBConverter",
  input    = cms.string("db"),
  dtLabel  = cms.string(""),
  cscLabel = cms.string(""),
  shiftErr = cms.double(1000.),
  angleErr = cms.double(6.28),
  getAPEs  = cms.bool(True),
  output   = cms.string("db")
)

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDBSetup,
  connect = cms.string("sqlite_file:%s" % outputFile),
  toPut = cms.VPSet(
    cms.PSet(record = cms.string("DTAlignmentRcd"),       tag = cms.string("DTAlignmentRcd")),
    cms.PSet(record = cms.string("DTAlignmentErrorRcd"),  tag = cms.string("DTAlignmentErrorRcd")),
    cms.PSet(record = cms.string("CSCAlignmentRcd"),      tag = cms.string("CSCAlignmentRcd")),
    cms.PSet(record = cms.string("CSCAlignmentErrorRcd"), tag = cms.string("CSCAlignmentErrorRcd"))
  )
)

process.Path = cms.Path(process.MuonGeometryDBConverter)
