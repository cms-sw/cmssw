import FWCore.ParameterSet.Config as cms

# single high-pT muon skim sequence

HighPtMuonSelection = "(isTrackerMuon || isGlobalMuon) && abs(eta) <= 2.4 && pt > 10."

highPtMuonSelectorForMuonIon = cms.EDFilter("PATMuonRefSelector",
    src = cms.InputTag("slimmedMuons"),
    cut = cms.string(HighPtMuonSelection),
    filter = cms.bool(True)
    )

highPtMuonCountFilterForMuonIon = cms.EDFilter("MuonRefPatCount",
    src = cms.InputTag("slimmedMuons"),
    cut = cms.string(HighPtMuonSelection),
    minNumber = cms.uint32(1)
    )

HighPtMuonIonSkimSequence = cms.Sequence(
    highPtMuonSelectorForMuonIon *
    highPtMuonCountFilterForMuonIon
    )


# loose dimuon skim sequence

LooseMuonSelection = "(isTrackerMuon || isGlobalMuon) && ((abs(eta) <= 1.0 && pt > 3.3) || (1.0 < abs(eta) <= 2.4 && pt > 1.0))"

looseMuonSelectorForMuonIon = cms.EDFilter("PATMuonRefSelector",
    src = cms.InputTag("slimmedMuons"),
    cut = cms.string(LooseMuonSelection),
    filter = cms.bool(True)
    )

looseMuonCountFilterForMuonIon = cms.EDFilter("MuonRefPatCount",
    src = cms.InputTag("slimmedMuons"),
    cut = cms.string(LooseMuonSelection),
    minNumber = cms.uint32(2)
    )


dimuonMassCutForMuonIon = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string("mass > 2.4"),
    decay = cms.string("looseMuonSelectorForMuonIon looseMuonSelectorForMuonIon")
    )

dimuonCountFilterForMuonIon = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("dimuonMassCutForMuonIon"),
    minNumber = cms.uint32(1)
    )

# dimuon skim sequence
DimuonIonSkimSequence = cms.Sequence(
    looseMuonSelectorForMuonIon *
    looseMuonCountFilterForMuonIon *
    dimuonMassCutForMuonIon *
    dimuonCountFilterForMuonIon
    )
