import os
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
process = cms.Process('PCL',Run2_2017)

# ----------------------------------------------------------------------
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.FwkReport.reportEvery = 10000
process.MessageLogger.HLTrigReport=dict()
process.MessageLogger.L1GtTrigReport=dict()
process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound'),
    wantSummary = cms.untracked.bool(True)
)

# -- Conditions
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '92X_dataRun2_Express_v8', '')

# -- Input files
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    "/store/express/Run2017F/ExpressPhysics/FEVT/Express-v1/000/305/366/00000/863EC350-6EB6-E711-8EAD-02163E019B61.root",
    "/store/express/Run2017F/ExpressPhysics/FEVT/Express-v1/000/305/366/00000/B6268B1F-6FB6-E711-A46C-02163E01439D.root",
    ),
    #lumisToProcess = cms.untracked.VLuminosityBlockRange("305366:1-305366:1"),
)


# -- number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import siPixelDigis
process.siPixelDigis = siPixelDigis.clone()
process.siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")

process.siPixelStatusProducer = cms.EDProducer("SiPixelStatusProducer",
    SiPixelStatusProducerParameters = cms.PSet(
        badPixelFEDChannelCollections = cms.VInputTag(cms.InputTag('siPixelDigis')),
        pixelClusterLabel = cms.untracked.InputTag("siPixelClusters::RECO"),
        monitorOnDoubleColumn = cms.untracked.bool(False),
        resetEveryNLumi = cms.untracked.int32( 1 )
    )
)

process.ALCARECOStreamSiPixelCalZeroBias = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('SiPixelCalZeroBias.root'),
    outputCommands = cms.untracked.vstring('drop *',
        'keep *_siPixelStatusProducer_*_*',
    )
)

process.p = cms.Path(process.siPixelDigis*process.siPixelStatusProducer)
process.end = cms.EndPath(process.ALCARECOStreamSiPixelCalZeroBias)

process.schedule = cms.Schedule(process.p,process.end)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

