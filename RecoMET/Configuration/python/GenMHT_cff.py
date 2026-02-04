import FWCore.ParameterSet.Config as cms

from RecoMET.METProducers.genMht_cff import genMht


process = cms.Process("SOMEPROCESS")
process.genMht = genMht

genMHTTask = cms.Task(process.genMht)
genMHT = cms.Sequence(genMHTTask)

process.p = cms.Path(genMHT)


# genMHTTask = cms.Task(genMht,)
# genMHT = cms.Sequence(genMHTTask)

