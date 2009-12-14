import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDigitizerTest")
#untracked PSet maxEvents = {untracked int32 input = 100}
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

#include "SimMuon/CSCDigitizer/data/muonCSCDbConditions.cfi"
#replace muonCSCDigis.stripConditions = "Database"
#replace muonCSCDigis.strips.ampGainSigma = 0.
#replace muonCSCDigis.strips.peakTimeSigma = 0.
#replace muonCSCDigis.strips.doNoise = false
#replace muonCSCDigis.wires.doNoise = false
#replace muonCSCDigis.strips.doCrosstalk = false
process.load("CalibMuon.Configuration.CSC_FakeDBConditions_cff")

#   include "CalibMuon/Configuration/data/CSC_FrontierConditions.cff"
#   replace cscConditions.toGet =  {
#        { string record = "CSCDBGainsRcd"
#          string tag = "CSCDBGains_ideal"},
#        {string record = "CSCNoiseMatrixRcd"
#          string tag = "CSCNoiseMatrix_ideal"},
#        {string record = "CSCcrosstalkRcd"
#          string tag = "CSCCrosstalk_ideal"},
#        {string record = "CSCPedestalsRcd"
#         string tag = "CSCPedestals_ideal"}
#    }
process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")
process.load("EventFilter.CSCRawToDigi.cscFrontierCablingUnpck_cff")

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('file:simevent.root')
)


process.DQMStore = cms.Service("DQMStore")

process.load("SimMuon.CSCDigitizer.cscDigiDump_cfi")

process.muonCSCDigis.InputObjects = "rawDataCollector"
process.p1 = cms.Path(process.muonCSCDigis)

