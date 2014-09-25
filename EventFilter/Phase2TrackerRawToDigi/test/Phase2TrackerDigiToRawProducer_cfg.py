import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("DigiToRaw")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring( 'root://xrootd.ba.infn.it/'+sys.argv[-1])
)

process.load('Configuration.Geometry.GeometryExtendedPhase2TkBEReco_cff')
process.load('DummyCablingTxt_cfi')
process.load('EventFilter.Phase2TrackerRawToDigi.Phase2TrackerDigiToRawProducer_cfi')

process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('digi2raw.root'),
    outputCommands = cms.untracked.vstring(
      'drop *',
#      'keep *_Phase2TrackerDigiToRawProducer_*_*',
      'keep *_siPixelClusters_*_*'
      )
    )


process.p = cms.Path(process.Phase2TrackerDigiToRawProducer)

process.e = cms.EndPath(process.out)
