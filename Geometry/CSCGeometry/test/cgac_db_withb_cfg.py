# Configuration file to run CSCGACwithB
# printing table of chamber information.
# Tim Cox 11.03.2011 - geometry from db & then aligned
# (cf. cgacwithb_cfg.py which uses ideal geometry)

import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryAsChambersWithB")

# To access the FULL geometry requires
# ====================================
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
# Any 'MC' tag is always ideal !BEWARE!
##process.GlobalTag.globaltag = 'MC_42_V6::All'
process.GlobalTag.globaltag = 'START42_V6::All'

# But all I need for CSC geometry tests is a subset of the above
# ==============================================================
##process.load('GeometryReaders.XMLIdealGeometryESSource.cmsGeometryDB_cff')
##process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')
##process.load('Geometry.CSCGeometryBuilder.cscGeometryDB_cfi')
##process.load('Geometry.CSCGeometryBuilder.idealForDigiCscGeometryDB_cff')
# note: the following is the standard entry point for condition records (necessary for CSC geometry)
##process.load('Configuration.StandardSequences.CondDBESSource_cff')
# Any 'MC' tag is always ideal !BEWARE!
##process.GlobalTag.globaltag = 'MC_42_V6::All'
##process.GlobalTag.globaltag = 'START42_V6::All'

# Load magnetic field (since this executable uses it)
# ===================================================
process.load("Configuration.StandardSequences.MagneticField_cff")

# Basic source & masxEvents
# =========================
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# MessageLogger - why even try?!
# ==============================
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules.append('CSCGeometryESModule')
process.MessageLogger.cout = cms.untracked.PSet(
   threshold = cms.untracked.string('DEBUG'),
   default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
   CSCGeometry = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
   CSCGeometryBuilder = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

process.producer = cms.EDAnalyzer("CSCGACwithB")

process.p1 = cms.Path(process.producer)

