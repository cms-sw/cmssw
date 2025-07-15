import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

from IOPool.Input.modules import PoolSource

process.source = PoolSource(fileNames = ["file:prov.root", "file:prov_extra.root"])

from FWCore.Modules.modules import ProvenanceCheckerOutputModule, AsciiOutputModule
process.out = ProvenanceCheckerOutputModule()
process.prnt = AsciiOutputModule(verbosity = 2, allProvenance=True)

process.e = cms.EndPath(process.out+process.prnt)
