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
GlobalTag.globaltag = "IDEAL_30X::All" 

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

# ----- Original RawToDigiToClusters chain -----

# DigiToRaw (orig) ### WARNING: default for cfi should be migrated to new once validated!!!
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
SiStripDigiToRaw.FedReadoutMode = 'ZERO_SUPPRESSED'
SiStripDigiToRaw.InputModuleLabel = 'DigiSource'
SiStripDigiToRaw.InputDigiLabel = ''
SiStripDigiToRaw.UseFedKey = False

# RawToDigi (orig)
siStripDigis = cms.EDProducer(
    "OldSiStripRawToDigiModule",
    ProductLabel =  cms.untracked.string('SiStripDigiToRaw'),
    UseFedKey = cms.untracked.bool(False),
    )

# Clusterizer (orig)
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
siStripClusters.DigiProducersList = cms.VPSet(
    cms.PSet(
    DigiLabel = cms.string('ZeroSuppressed'),
    DigiProducer = cms.string('siStripDigis')
    )
    )

# Clusterizer (new)
from RecoLocalTracker.SiStripClusterizer.SiStripClusterProducer_cfi import *
siStripClusterProducer.ProductLabel = cms.InputTag('siStripDigis:ZeroSuppressed')
siStripClusterProducer.DetSetVectorNew = True


# ----- New RawToClusters chain -----


# DigiToRaw (new) ### WARNING: default for cfi should be migrated to new once validated!!!
newSiStripDigiToRaw = cms.EDProducer(
    "SiStripDigiToRawModule",
    InputModuleLabel = cms.string('simSiStripDigis'),
    InputDigiLabel = cms.string('ZeroSuppressed'),
    FedReadoutMode = cms.untracked.string('ZERO_SUPPRESSED'),
    UseFedKey = cms.untracked.bool(False)
    )

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


# DigiToRaw (old) ### WARNING: default for cfi should be migrated to new once validated!!!
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
oldSiStripDigiToRaw = SiStripDigiToRaw.clone()
oldSiStripDigiToRaw.FedReadoutMode = 'ZERO_SUPPRESSED'
oldSiStripDigiToRaw.InputModuleLabel = 'DigiSource'
oldSiStripDigiToRaw.InputDigiLabel = ''
oldSiStripDigiToRaw.UseFedKey = False

# RawToClusters (old)
oldSiStripRawToClustersFacility = cms.EDProducer(
    "OldSiStripRawToClusters",
    SiStripClusterization,
    ProductLabel = cms.InputTag('oldSiStripDigiToRaw')
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


# Cluster Validator (old-to-orig)
from EventFilter.SiStripRawToDigi.test.SiStripClusterValidator_cfi import *
oldValidateSiStripClusters = ValidateSiStripClusters.clone()
oldValidateSiStripClusters.Collection1 = cms.untracked.InputTag("siStripClusterProducer")
oldValidateSiStripClusters.Collection2 = cms.untracked.InputTag("oldSiStripClustersDSV")
oldValidateSiStripClusters.DetSetVectorNew = True

# Cluster Validator (new-to-orig)
newValidateSiStripClusters = ValidateSiStripClusters.clone()
newValidateSiStripClusters.Collection1 = cms.untracked.InputTag("siStripClusterProducer")
newValidateSiStripClusters.Collection2 = cms.untracked.InputTag("siStripClustersDSV")
newValidateSiStripClusters.DetSetVectorNew = True

# Cluster Validator (new-to-old)
testValidateSiStripClusters = ValidateSiStripClusters.clone()
testValidateSiStripClusters.Collection1 = cms.untracked.InputTag("oldSiStripClustersDSV")
testValidateSiStripClusters.Collection2 = cms.untracked.InputTag("siStripClustersDSV")
testValidateSiStripClusters.DetSetVectorNew = True


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

orig = cms.Sequence(
    SiStripDigiToRaw *
    siStripDigis *
    siStripClusterProducer
    )

old = cms.Sequence(
    oldSiStripDigiToRaw *
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

test = cms.Sequence(
    testValidateSiStripClusters
    )

e = cms.EndPath( output )
s = cms.Sequence( dummySiStripDigiToRaw * orig * old * new * test )
