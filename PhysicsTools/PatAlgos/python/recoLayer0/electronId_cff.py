import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.electronIdSequence_cff import *

patElectronIds = cms.EDFilter("CandManyValueMapsSkimmerFloat",
    collection = cms.InputTag("allLayer0Electrons"),
    backrefs   = cms.InputTag("allLayer0Electrons"),
    associations = cms.VInputTag(
        cms.InputTag("eidRobustLoose"),
        cms.InputTag("eidRobustTight"),
        cms.InputTag("eidLoose"),
        cms.InputTag("eidTight"),
    ),
    failSilently = cms.untracked.bool(False),
)

patElectronId = cms.Sequence(
    eIdSequence * patElectronIds
)
