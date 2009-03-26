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


# ----- Common components -----


# Digi Source (orig, old and new)
from EventFilter.SiStripRawToDigi.test.SiStripTrivialDigiSource_cfi import *
DigiSource.FedRawDataMode = False
DigiSource.UseFedKey = False

# DigiToRaw (orig, old and new) 
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
SiStripDigiToRaw.UseFedKey = False


# ----- Original RawToDigiToClusters chain -----


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


# RawToClusters (new)
from EventFilter.SiStripRawToDigi.SiStripRawToClusters_cfi import *
SiStripRawToClustersFacility.ProductLabel = cms.InputTag("SiStripDigiToRaw")

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
    siStripDigis *
    siStripClusterProducer
    )

old = cms.Sequence(
    oldSiStripRawToClustersFacility *
    oldSiStripRoI *
    oldSiStripClustersDSV *
    oldValidateSiStripClusters
    )

new = cms.Sequence(
    SiStripRawToClustersFacility *
    SiStripRoI *
    siStripClustersDSV *
    newValidateSiStripClusters
    )

test = cms.Sequence(
    testValidateSiStripClusters
    )

e = cms.EndPath( output )
s = cms.Sequence( SiStripDigiToRaw * orig * old * new * test )
