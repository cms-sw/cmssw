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
GlobalTag.globaltag = "MC_31X_V6::All" 

# Region cabling
from Configuration.StandardSequences.Geometry_cff import *
from Configuration.StandardSequences.MagneticField_cff import *
from CalibTracker.SiStripESProducers.SiStripRegionConnectivity_cfi import *

# Digi Source (common)
from EventFilter.SiStripRawToDigi.test.SiStripTrivialDigiSource_cfi import *
DigiSource.FedRawDataMode = False
DigiSource.UseFedKey = False

# DigiToRaw (dummy, not used, for timing purposes only)
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
dummySiStripDigiToRaw = SiStripDigiToRaw.clone()


# ----- Reference RawToDigiToClusters chains -----

# Old DigiToRaw and RawToDigi
oldSiStripDigiToRaw = cms.EDProducer(
    "OldSiStripDigiToRawModule",
    InputDigis = cms.InputTag("DigiSource", ""),
    FedReadoutMode = cms.untracked.string('ZERO_SUPPRESSED'),
    PacketCode = cms.untracked.string('ZERO_SUPPRESSED'),
    UseFedKey = cms.untracked.bool(False)
    )
oldSiStripDigis = cms.EDProducer(
    "OldSiStripRawToDigiModule",
    ProductLabel      = cms.InputTag('oldSiStripDigiToRaw'),
    AppendedBytes     = cms.untracked.int32(0),
    UseDaqRegister    = cms.bool(False),
    UseFedKey         = cms.untracked.bool(False),
    UnpackBadChannels = cms.bool(False),
    TriggerFedId      = cms.untracked.int32(0)
    #FedEventDumpFreq  = cms.untracked.int32(0),
    #FedBufferDumpFreq = cms.untracked.int32(0),
    )

# New DigiToRaw and RawToDigi
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
SiStripDigiToRaw.InputDigis = cms.InputTag("DigiSource", "ZeroSuppressed")
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
siStripDigis.ProductLabel = 'SiStripDigiToRaw'

# Clusterizer (reference "new")
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
siStripClusters.DigiProducersList = cms.VInputTag(cms.InputTag('siStripDigis:ZeroSuppressed'))


# ----- New RawToClusters chain -----

# DigiToRaw (new)
newSiStripDigiToRaw = SiStripDigiToRaw.clone()

# RawToClusters (new)
from EventFilter.SiStripRawToDigi.SiStripRawToClusters_cfi import *
SiStripRawToClustersFacility.ProductLabel = cms.InputTag("newSiStripDigiToRaw")

# Regions Of Interest (new)
from EventFilter.SiStripRawToDigi.SiStripRawToClustersRoI_cfi import *
SiStripRoI.SiStripLazyGetter = cms.InputTag("SiStripRawToClustersFacility")

# Clusters DSV Builder (new)
from EventFilter.SiStripRawToDigi.test.SiStripClustersDSVBuilder_cfi import *
siStripClustersDSV.SiStripLazyGetter = cms.InputTag("SiStripRawToClustersFacility")
siStripClustersDSV.SiStripRefGetter = cms.InputTag("SiStripRoI")
siStripClustersDSV.DetSetVectorNew = True


# ----- Validators -----

from EventFilter.SiStripRawToDigi.test.SiStripClusterValidator_cfi import *

# Cluster Validator (new-to-reference)
newValidateSiStripClusters = ValidateSiStripClusters.clone()
newValidateSiStripClusters.Collection1 = cms.untracked.InputTag("siStripClusters")
newValidateSiStripClusters.Collection2 = cms.untracked.InputTag("siStripClustersDSV")
newValidateSiStripClusters.DetSetVectorNew = True


# ----- Sequences and Paths -----


# PoolOutput
output = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('output.root'),
    outputCommands = cms.untracked.vstring(
    'drop *',
    'keep SiStrip*_simSiStripDigis_*_*', # (to drop SimLinks)
    'keep *_*_*_DigiToRawToClusters'
    )
    )


reference_new = cms.Sequence(
    SiStripDigiToRaw *
    siStripDigis *
    siStripClusters
    )

new = cms.Sequence(
    newSiStripDigiToRaw *
    SiStripRawToClustersFacility *
    SiStripRoI *
    siStripClustersDSV *
    newValidateSiStripClusters
    )

#test = cms.Sequence(
#    testValidateSiStripClusters
#    )

e = cms.EndPath( output )
s = cms.Sequence( dummySiStripDigiToRaw * reference_new * new ) #* test )
