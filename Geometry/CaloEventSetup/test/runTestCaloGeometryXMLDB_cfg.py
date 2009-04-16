# The following comments couldn't be translated into the new config version:

# timing and memory checks

import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

# Magnetic field full setup
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# Geometry - overkill, used for test/completeness 
#            includes Sim, Digi & Reco files as far
#            as I can tell.

#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.CaloEventSetup.xmlsqlitefile")

# Calo geometry service model

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

# Ecal TT mapping
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")



process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(4)
)
process.source = cms.Source("EmptySource")

process.etta = cms.EDFilter("dumpEcalTrigTowerMapping")

process.ctgw = cms.EDFilter("testEcalGetWindow")

process.cga = cms.EDAnalyzer("CaloGeometryAnalyzer",
    fullEcalDump = cms.untracked.bool(True)
)

process.mfa = cms.EDFilter("testMagneticField")

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.TFileService = cms.Service("TFileService", fileName = cms.string('calogeom.root') )

process.p1 = cms.Path(process.etta*process.ctgw*process.cga*process.mfa)


