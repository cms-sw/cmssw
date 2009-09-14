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
    InputModuleLabel = cms.string('DigiSource'),
    InputDigiLabel = cms.string(''),
    FedReadoutMode = cms.untracked.string('ZERO_SUPPRESSED'),
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

# Clusterizer (reference "old")
from RecoLocalTracker.SiStripClusterizer.SiStripClusterProducer_cfi import *
siStripClusterProducer.ProductLabel = cms.InputTag('oldSiStripDigis:ZeroSuppressed')
siStripClusterProducer.DetSetVectorNew = True

# New DigiToRaw and RawToDigi
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
SiStripDigiToRaw.InputModuleLabel = 'DigiSource'
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


# ----- Old RawToClusters chain -----


# RawToClusters (old)
oldSiStripRawToClustersFacility = cms.EDProducer(
    "OldSiStripRawToClusters",
    SiStripClusterization,
    ProductLabel = cms.InputTag('SiStripDigiToRaw')
    )

# Regions Of Interest (old)
oldSiStripRoI = SiStripRoI.clone()
oldSiStripRoI.SiStripLazyGetter = cms.InputTag("oldSiStripRawToClustersFacility")

# Clusters DSV Builder (old)
oldSiStripClustersDSV = siStripClustersDSV.clone()
oldSiStripClustersDSV.SiStripLazyGetter = cms.InputTag("oldSiStripRawToClustersFacility")
oldSiStripClustersDSV.SiStripRefGetter = cms.InputTag("oldSiStripRoI")
oldSiStripClustersDSV.DetSetVectorNew = True


# ----- Validators -----


# Cluster Validator (old-to-reference)
from EventFilter.SiStripRawToDigi.test.SiStripClusterValidator_cfi import *
oldValidateSiStripClusters = ValidateSiStripClusters.clone()
oldValidateSiStripClusters.Collection1 = cms.untracked.InputTag("siStripClusterProducer")
oldValidateSiStripClusters.Collection2 = cms.untracked.InputTag("oldSiStripClustersDSV")
oldValidateSiStripClusters.DetSetVectorNew = True

# Cluster Validator (new-to-reference)
newValidateSiStripClusters = ValidateSiStripClusters.clone()
newValidateSiStripClusters.Collection1 = cms.untracked.InputTag("siStripClusters")
newValidateSiStripClusters.Collection2 = cms.untracked.InputTag("siStripClustersDSV")
newValidateSiStripClusters.DetSetVectorNew = True

## Cluster Validator (new-to-old)
#testValidateSiStripClusters = ValidateSiStripClusters.clone()
#testValidateSiStripClusters.Collection1 = cms.untracked.InputTag("oldSiStripClustersDSV")
#testValidateSiStripClusters.Collection2 = cms.untracked.InputTag("siStripClustersDSV")
#testValidateSiStripClusters.DetSetVectorNew = True


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

reference_old = cms.Sequence(
    oldSiStripDigiToRaw *
    oldSiStripDigis *
    siStripClusterProducer
    )


reference_new = cms.Sequence(
    SiStripDigiToRaw *
    siStripDigis *
    siStripClusters
    )

old = cms.Sequence(
    SiStripDigiToRaw *
    oldSiStripRawToClustersFacility *
    oldSiStripRoI *
    oldSiStripClustersDSV *
    oldValidateSiStripClusters
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
s = cms.Sequence( dummySiStripDigiToRaw * reference_old * reference_new * old * new ) #* test )
