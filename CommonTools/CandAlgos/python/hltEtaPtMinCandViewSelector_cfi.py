import FWCore.ParameterSet.Config as cms

hltEtaPtMinCandViewSelector = cms.EDFilter("EtaPtMinCandViewSelector",
   src    = cms.InputTag("hltCollection"),
   ptMin  = cms.double(-1.0),
   etaMin = cms.double(-1e125),
   etaMax = cms.double(+1e125)
)
