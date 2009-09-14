import FWCore.ParameterSet.Config as cms

Mode = str("ZS")    # Options: "ZS", "VR", "PR", "FK"
Write = bool(False) # Write output to disk

process = cms.Process("DigiToRawToDigi")

# ---- Data source ----
process.source = cms.Source(
   "PoolSource",
   fileNames = cms.untracked.vstring(
   '/store/relval/CMSSW_3_1_1/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/ECAD7ED7-966B-DE11-B4FE-000423D99CEE.root'
    )
)

# ---- Services ----
process.load("DQM.SiStripCommon.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring()
process.Timing = cms.Service("Timing")
process.Tracer = cms.Service(
    "Tracer",
    sourceSeed = cms.untracked.string("$$")
    )

# ---- Conditions ----
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "MC_31X_V2::All"

# ---- DigiToRaw ----
process.load("EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi")
process.SiStripDigiToRaw.InputModuleLabel = 'simSiStripDigis'
process.SiStripDigiToRaw.InputDigiLabel = 'ZeroSuppressed'

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
    process.SiStripDigiToRaw.FedReadoutMode = 'ZERO_SUPPRESSED'
elif Mode == str("VR") :
    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
    process.SiStripDigiToRaw.FedReadoutMode = 'VIRGIN_RAW'
    process.DigiValidator.TagCollection2 = "testSiStripDigis:VirginRaw"
    process.DigiValidator.RawCollection2 = True
elif Mode == str("PR") :
    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
    process.SiStripDigiToRaw.FedReadoutMode = 'PROCESSED_RAW'
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

