import FWCore.ParameterSet.Config as cms

Mode = str("ZS")    # Options: "ZS", "VR", "PR", "FK"
Write = bool(False) # Write output to disk

process = cms.Process("DigiToRawToClusters")

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

# ---- Region cabling ----
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('CalibTracker.SiStripESProducers.SiStripRegionConnectivity_cfi')

# ---- Reference clusters ----
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
process.referenceSiStripClusters = siStripClusters.clone()
process.referenceSiStripClusters.DigiProducersList = cms.VInputTag(cms.InputTag('simSiStripDigis:ZeroSuppressed'))

# ---- DigiToRaw ----
process.load("EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi")
process.SiStripDigiToRaw.InputDigis = cms.InputTag('simSiStripDigis', 'ZeroSuppressed')

# ---- RawToClusters ----
process.load('EventFilter.SiStripRawToDigi.SiStripRawToClusters_cfi')
process.SiStripRawToClustersFacility.ProductLabel = cms.InputTag("SiStripDigiToRaw")
process.load('EventFilter.SiStripRawToDigi.SiStripRawToClustersRoI_cfi')
process.SiStripRoI.SiStripLazyGetter = cms.InputTag("SiStripRawToClustersFacility")
process.load('EventFilter.SiStripRawToDigi.test.SiStripClustersDSVBuilder_cfi')
process.siStripClustersDSV.SiStripLazyGetter = cms.InputTag("SiStripRawToClustersFacility")
process.siStripClustersDSV.SiStripRefGetter = cms.InputTag("SiStripRoI")
process.siStripClustersDSV.DetSetVectorNew = True
process.SiStripRawToClusters = cms.Sequence( process.SiStripRawToClustersFacility * process.SiStripRoI * process.siStripClustersDSV )

# ---- Validation ----
process.load('EventFilter.SiStripRawToDigi.test.SiStripClusterValidator_cfi')
process.ValidateSiStripClusters.Collection1 = cms.untracked.InputTag("referenceSiStripClusters")
process.ValidateSiStripClusters.Collection2 = cms.untracked.InputTag("siStripClustersDSV")
process.ValidateSiStripClusters.DetSetVectorNew = True

# ----- FedReadoutMode -----
if Mode == str("ZS") :
    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
    process.SiStripDigiToRaw.FedReadoutMode = cms.string('ZERO_SUPPRESSED')
    process.SiStripDigiToRaw.PacketCode = cms.string('ZERO_SUPPRESSED')
elif Mode == str("VR") :
    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
    process.SiStripDigiToRaw.FedReadoutMode = cms.string('VIRGIN_RAW')
    process.SiStripDigiToRaw.PacketCode = cms.string('VIRGIN_RAW')
elif Mode == str("PR") :
    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
    process.SiStripDigiToRaw.FedReadoutMode = cms.string('PROCESSED_RAW')
    process.SiStripDigiToRaw.PacketCode = cms.string('PROCESSED_RAW')
else :
    print "UNKNOWN FED READOUT MODE!"
    import sys
    sys.exit()

# ---- Sequence ----
process.p = cms.Path(
    process.referenceSiStripClusters *
    process.SiStripDigiToRaw *
    process.SiStripRawToClusters *
    process.ValidateSiStripClusters
    )

# ----- WriteToDisk -----
process.output = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('output.root'),
    outputCommands = cms.untracked.vstring(
    'drop *',
    'keep SiStrip*_simSiStripDigis_*_*', # (to drop SimLinks)
    'keep *_*_*_DigiToRawToClusters'
    )
    )
process.output.fileName = "DigiToRawToClusters"+Mode+".root"
if Write == bool(True) :
    process.e = cms.EndPath( process.output )
else :
    print "Event content not written to disk!" 

