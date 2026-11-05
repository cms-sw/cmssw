# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# A later job that reads the GEN-only graph and writes the metadata from the process history.
# With the argument "shadow", the job also runs a module under the graph producer label that
# makes no graph; the metadata must still describe the producer of the graph in the input file.
import sys

import FWCore.ParameterSet.Config as cms

from PhysicsTools.TruthInfo.customiseTruthMetadata import addTruthMetadata

shadow = "shadow" in sys.argv[1:]
outputFile = "genOnlyGraphMetaShadow.root" if shadow else "genOnlyGraphMeta.root"

process = cms.Process("META")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring("file:genOnlyGraph.root"),
                            inputCommands=cms.untracked.vstring("keep *", "drop *_truthGraphMetadata_*_*"))
process.out = cms.OutputModule("PoolOutputModule", fileName=cms.untracked.string(outputFile),
                               outputCommands=cms.untracked.vstring("drop *", "keep *_truthGenOnly*_*_*"))
process.e = cms.EndPath(process.out)
addTruthMetadata(process, graphProducer="truthGenOnlyLogicalGraphProducer",
                 rawGraphProducer="truthGenOnlyGraphProducer")
if shadow:
    process.truthGenOnlyLogicalGraphProducer = cms.EDAnalyzer("EventContentAnalyzer", verbose=cms.untracked.bool(False))
    process.shadow = cms.Path(process.truthGenOnlyLogicalGraphProducer)
