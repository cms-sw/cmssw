import FWCore.ParameterSet.Config as cms

Mode = str("ZS")    # Options: "ZS", "VR", "PR", "FK"
Write = bool(False) # Write output to disk

process = cms.Process("DigiToRawToDigi")

# ---- Data source ----
process.source = cms.Source(
   "PoolSource",
   fileNames = cms.untracked.vstring(
   '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0015/FA48FEA0-191E-DF11-9B68-003048679076.root'
    )
   )

# ---- Services ----
process.load("DQM.SiStripCommon.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring()
process.Timing = cms.Service("Timing")
process.Tracer = cms.Service("Tracer")

# ---- Conditions ----
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "MC_3XY_V21::All"

# ---- DigiToRaw ----
process.load("EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi")
process.SiStripDigiToRaw.InputDigis = cms.InputTag('simSiStripDigis', "ZeroSuppressed")

# ---- RawToDigi ----
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
process.testSiStripDigis = siStripDigis.clone()
process.testSiStripDigis.ProductLabel = 'SiStripDigiToRaw'

# ---- Validation ----
process.load('EventFilter.SiStripRawToDigi.test.SiStripDigiValidator_cfi')
process.DigiValidator.TagCollection1 = "simSiStripDigis:ZeroSuppressed"
process.DigiValidator.TagCollection2 = "testSiStripDigis:ZeroSuppressed"

# ----- FedReadoutMode -----
if Mode == str("ZS") :
    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
    process.SiStripDigiToRaw.FedReadoutMode = cms.string('ZERO_SUPPRESSED')
    process.SiStripDigiToRaw.PacketCode = cms.string('ZERO_SUPPRESSED')
elif Mode == str("VR") :
    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
    process.SiStripDigiToRaw.FedReadoutMode = cms.string('VIRGIN_RAW')
    process.SiStripDigiToRaw.PacketCode = cms.string('VIRGIN_RAW')
    process.DigiValidator.TagCollection2 = "testSiStripDigis:VirginRaw"
    process.DigiValidator.RawCollection2 = True
elif Mode == str("PR") :
    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
    process.SiStripDigiToRaw.FedReadoutMode = cms.string('PROCESSED_RAW')
    process.SiStripDigiToRaw.PacketCode = cms.string('PROCESSED_RAW')
    process.DigiValidator.TagCollection2 = "testSiStripDigis:ProcessedRaw"
    process.DigiValidator.RawCollection2 = True
else :
    print "UNKNOWN FED READOUT MODE!"
    import sys
    sys.exit()

# ---- Sequence ----
process.p = cms.Path(
    process.SiStripDigiToRaw *
    process.testSiStripDigis *
    process.DigiValidator
    )

# ----- WriteToDisk -----
process.output = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('output.root'),
    outputCommands = cms.untracked.vstring(
    'drop *',
    'keep SiStrip*_simSiStripDigis_*_*', # (to drop SimLinks)
    'keep *_*_*_DigiToRawToDigi'
    )
    )
process.output.fileName = "DigiToRawToDigi"+Mode+".root"
if Write == bool(True) :
    process.e = cms.EndPath( process.output )
else :
    print "Event content not written to disk!" 

