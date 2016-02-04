import FWCore.ParameterSet.Config as cms

hltL2MuonPoiting = cms.EDFilter("L2MuonPoiting",
    maxZ = cms.double(293.5),
    #ecal
    radius = cms.double(129.0),
    PropagatorName = cms.string('SteppingHelixPropagator'),
    SALabel = cms.string('cosmicMuons')
)


