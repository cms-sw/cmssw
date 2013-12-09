import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi import selectedPatElectrons
isolatedPatElectrons = selectedPatElectrons.clone(src="selectedPatElectrons", cut="pt>10 & abs(eta)<2.5 & (trackIso+caloIso)/pt<5")

from PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi import selectedPatMuons
isolatedPatMuons = selectedPatMuons.clone(src="selectedPatMuons", cut="pt>10 & abs(eta)<2.5 & (trackIso+caloIso)/pt<5")

customSelection = cms.Sequence(
    isolatedPatElectrons *isolatedPatMuons
    )
