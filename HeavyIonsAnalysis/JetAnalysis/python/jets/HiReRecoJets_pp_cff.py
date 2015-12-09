import FWCore.ParameterSet.Config as cms
from RecoJets.JetProducers.ak5PFJets_cfi import ak5PFJets
ak5PFJets.doAreaFastjet = True
ak5PFJets.jetPtMin = 1
from RecoJets.JetProducers.ak5CaloJets_cfi import ak5CaloJets
ak5CaloJets.doAreaFastjet = True
ak5CaloJets.jetPtMin = 1

ak1PFJets = ak5PFJets.clone(rParam = 0.1)
ak2PFJets = ak5PFJets.clone(rParam = 0.2)
ak3PFJets = ak5PFJets.clone(rParam = 0.3)
ak4PFJets = ak5PFJets.clone(rParam = 0.4)
ak6PFJets = ak5PFJets.clone(rParam = 0.6)

ak1CaloJets = ak5CaloJets.clone(rParam = 0.1)
ak2CaloJets = ak5CaloJets.clone(rParam = 0.2)
ak3CaloJets = ak5CaloJets.clone(rParam = 0.3)
ak4CaloJets = ak5CaloJets.clone(rParam = 0.4)
ak6CaloJets = ak5CaloJets.clone(rParam = 0.6)

ppReRecoPFJets = cms.Sequence(
ak1PFJets
+
ak2PFJets
+
ak3PFJets
+
ak4PFJets
+
ak5PFJets
+
ak6PFJets
)

ppReRecoCaloJets = cms.Sequence(
ak1CaloJets
+
ak2CaloJets
+
ak3CaloJets
+
ak4CaloJets
+
ak5CaloJets
+
ak6CaloJets
)
