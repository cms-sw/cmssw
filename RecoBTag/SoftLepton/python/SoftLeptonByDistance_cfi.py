import FWCore.ParameterSet.Config as cms

# Do not remove - needed for ConfDB parsing for HLT!

softLeptonByDistance = cms.ESProducer("LeptonTaggerByDistanceESProducer",
    distance = cms.double(0.5)
)
# foo bar baz
# l0OQaQa3nVQom
# lyhukFwMPiRWB
