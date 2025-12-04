import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.maxEvents.input = 10

from FWCore.Modules.modules import EmptySource

process.source = EmptySource()

from FWCore.Integration.modules import ThingProducer

process.thing = ThingProducer()

from FWIO.RNTupleTempOutput.modules import RNTupleTempOutputModule

process.out = RNTupleTempOutputModule(fileName = "alias.root",
                               branchAliases = [cms.PSet(branch=cms.untracked.string("*_thing_*_*"), alias = cms.untracked.string("foo"))])

process.e = cms.EndPath(process.out, cms.Task(process.thing))
