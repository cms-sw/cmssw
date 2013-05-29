import FWCore.ParameterSet.Config as cms
#FIXME: It needs to be in the chain of Top projector: for example, src=pfNoElectron 
pfSelectedPhotons = cms.EDFilter(
    "GenericPFCandidateSelector",
    src = cms.InputTag("pfAllPhotons"),
    cut = cms.string("mva_nothing_gamma>0")
)




