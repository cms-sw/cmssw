import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("DigiToRaw")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring( 'root://xrootd.ba.infn.it/'+sys.argv[-1])
)

process.load('EventFilter.Phase2TrackerRawToDigi.Phase2TrackerDigiToRawProducer_cfi')

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myOutputFile.root')
)

process.p = cms.Path(process.Phase2TrackerDigiToRawProducer)

process.e = cms.EndPath(process.out)
