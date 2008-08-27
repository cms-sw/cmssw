#!/usr/bin/env pythona
import FWCore.ParameterSet.Config as cms

# process name
##############
process = cms.Process("GeometryTestTwo")

# empty input
#############
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# xml for endcap csc geometry
#############################
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
#process.load("Geometry.CSCGeometryBuilder.idealForDigiCscGeometry_cff")
#process.load("CalibMuon.Configuration.CSC_FakeConditions_cff")
process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")

# User defined modules
######################

process.cscmakesrlut = cms.EDAnalyzer("CSCMakeSRLUT",
	BinaryOutput = cms.untracked.bool(False),
	WriteGlobalEta = cms.untracked.bool(True),
	WriteGlobalPhi = cms.untracked.bool(True),
	WriteLocalPhi = cms.untracked.bool(True),
	Station = cms.untracked.int32(-1),
	Endcap = cms.untracked.int32(-1),
	Sector = cms.untracked.int32(-1),
	isTMB07 = cms.untracked.bool(True),
	lutParam = cms.PSet(
		UseMiniLUTs = cms.untracked.bool(True)
	)
)

process.cscmakeptlut = cms.EDAnalyzer("CSCMakePTLUT",
   BinaryOutput = cms.untracked.bool(False),
   lutParam = cms.PSet()
)

process.Path = cms.Path(process.cscmakesrlut)
#process.Path = cms.Path(process.cscmakeptlut)				
#process.Path = cms.Path(process.cscmakesrlut+process.cscmakeptlut)
