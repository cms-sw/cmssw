
ak1HiGenJetsCleaned = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak1HiGenJets') )

ak2HiGenJetsCleaned = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak2HiGenJets') )

ak3HiGenJetsCleaned = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak3HiGenJets') )

ak4HiGenJetsCleaned = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak4HiGenJets') )

ak5HiGenJetsCleaned = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak5HiGenJets') )

ak6HiGenJetsCleaned = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak6HiGenJets') )

ak7HiGenJetsCleaned = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak7HiGenJets') )

hiGenJets = cms.Sequence(
ak1HiGenJets
+
ak2HiGenJets
+
ak3HiGenJets
+
ak4HiGenJets
+
ak5HiGenJets
+
ak6HiGenJets
+
ak7HiGenJets
)
