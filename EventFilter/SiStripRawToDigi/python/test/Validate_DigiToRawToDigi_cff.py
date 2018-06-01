import FWCore.ParameterSet.Config as cms

# Services
from DQM.SiStripCommon.MessageLogger_cfi import *
MessageLogger.debugModules = cms.untracked.vstring()
Timing = cms.Service("Timing")
Tracer = cms.Service(
    "Tracer", 
    sourceSeed = cms.untracked.string("$$")
    )

# Conditions
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
GlobalTag.globaltag = "MC_31X_V2::All" 

# Digi Source (common)
from EventFilter.SiStripRawToDigi.test.SiStripTrivialDigiSource_cfi import *
DigiSource.FedRawDataMode = False
DigiSource.UseFedKey = False

# DigiToRaw (dummy, not used, for timing purposes only)
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
dummySiStripDigiToRaw = SiStripDigiToRaw.clone()

# DigiToRaw (old) ### WARNING: default for cfi should be migrated to new once validated!!!
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
oldSiStripDigiToRaw = cms.EDProducer(
    "OldSiStripDigiToRawModule",
    InputDigis = cms.InputTag("DigiSource", ""),
    FedReadoutMode = cms.untracked.string('ZERO_SUPPRESSED'),
    PacketCode = cms.untracked.string('ZERO_SUPPRESSED'),
    UseFedKey = cms.untracked.bool(False)
    )

# DigiToRaw (new) ### WARNING: default for cfi should be migrated to new once validated!!!
newSiStripDigiToRaw = cms.EDProducer(
    "SiStripDigiToRawModule",
    InputDigis = cms.InputTag("simSiStripDigis", "ZeroSuppressed"),
    FedReadoutMode = cms.string('ZERO_SUPPRESSED'),
    PacketCode = cms.string('ZERO_SUPPRESSED'),
    UseFedKey = cms.bool(False),
    UseWrongDigiType = cms.bool(False)
    )

# RawToDigi (new)
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
siStripDigis.ProductLabel = 'newSiStripDigiToRaw'
siStripDigis.UseFedKey = False

# RawToDigi (old)
oldSiStripDigis = cms.EDProducer(
    "OldSiStripRawToDigiModule",
    ProductLabel =  cms.InputTag('oldSiStripDigiToRaw'),
    UseFedKey = cms.untracked.bool(False),
    )

# Digi Validator (new)
from EventFilter.SiStripRawToDigi.test.SiStripDigiValidator_cfi import *
DigiValidator.TagCollection1 = "DigiSource"
DigiValidator.TagCollection2 = "siStripDigis:ZeroSuppressed"
DigiValidator.RawCollection1 = False
DigiValidator.RawCollection2 = False

# Digi Validator (old)
oldDigiValidator = DigiValidator.clone()
oldDigiValidator.TagCollection1 = "DigiSource"
oldDigiValidator.TagCollection2 = "oldSiStripDigis:ZeroSuppressed"
oldDigiValidator.RawCollection1 = False
oldDigiValidator.RawCollection2 = False

# Digi Validator (compare)
testDigiValidator = DigiValidator.clone()
testDigiValidator.TagCollection1 = "oldSiStripDigis:ZeroSuppressed"
testDigiValidator.TagCollection2 = "siStripDigis:ZeroSuppressed"
testDigiValidator.RawCollection1 = False
testDigiValidator.RawCollection2 = False

# PoolOutput
output = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('output.root'),
    outputCommands = cms.untracked.vstring(
    'drop *',
    'keep SiStrip*_simSiStripDigis_*_*', # (to drop SimLinks)
    'keep *_*_*_DigiToRawToDigi'
    )
    )

# Sequences and Paths
new = cms.Sequence(
    newSiStripDigiToRaw *
    siStripDigis *
    DigiValidator
    )

old = cms.Sequence(
    oldSiStripDigiToRaw *
    oldSiStripDigis *
    oldDigiValidator
    )

test = cms.Sequence(
    testDigiValidator
    )

s = cms.Sequence( dummySiStripDigiToRaw * old * new * test )

