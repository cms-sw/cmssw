# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# A later job that reads the GEN-only graph and writes the metadata from the process history.
import FWCore.ParameterSet.Config as cms

from PhysicsTools.TruthInfo.customiseTruthMetadata import addTruthMetadata

process = cms.Process("META")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring("file:genOnlyGraph.root"),
                            inputCommands=cms.untracked.vstring("keep *", "drop *_truthGraphMetadata_*_*"))
process.out = cms.OutputModule("PoolOutputModule", fileName=cms.untracked.string("genOnlyGraphMeta.root"),
                               outputCommands=cms.untracked.vstring("drop *", "keep *_truthGenOnly*_*_*"))
process.e = cms.EndPath(process.out)
addTruthMetadata(process, graphProducer="truthGenOnlyLogicalGraphProducer",
                 rawGraphProducer="truthGenOnlyGraphProducer")
