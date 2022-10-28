from __future__ import print_function
import FWCore.ParameterSet.Config as cms

from Configuration.AlCa.autoCond import autoCond

process = cms.Process("TEST")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.source = cms.Source("EmptyIOVSource",
                                lastValue = cms.uint64(3),
                                timetype = cms.string('runnumber'),
                                firstValue = cms.uint64(1),
                                interval = cms.uint64(1)
                            )

from CondCore.ESSources.GlobalTag import GlobalTag

# Prepare the list of globalTags
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

globalTag = GlobalTag(autoCond['run2_data'],"frontier://FrontierProd/CMS_CONDITIONS")

process.GlobalTag.connect = cms.string(globalTag.connect())
process.GlobalTag.globaltag = globalTag.gt()

print("Final connection string =", process.GlobalTag.connect)
print("Final globalTag =", process.GlobalTag.globaltag)

process.path = cms.Path()
