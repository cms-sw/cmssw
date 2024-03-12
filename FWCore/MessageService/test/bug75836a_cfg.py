import FWCore.ParameterSet.Config as cms
process = cms.Process("MERGE")

process.source = cms.Source("EmptySource")

process.thing = cms.EDProducer ("IntProducer")

process.o = cms.Path(process.thing)

# foo bar baz
# 1QZZquvU53oDh
# oVSB7P7vFiUYL
