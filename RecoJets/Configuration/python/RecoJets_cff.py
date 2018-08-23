import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4CaloJets_cfi import ak4CaloJets

from RecoJets.JetProducers.fixedGridRhoProducerFastjet_cfi import fixedGridRhoFastjetAllCalo

doPileup = cms.bool(True)

ak4CaloJetsPUCorr           =ak4CaloJets.clone           (doPUOffsetCorr = doPileup)
ak4CaloJets.doAreaFastjet = True

fixedGridRhoFastjetCentralCalo = fixedGridRhoFastjetAllCalo.clone(
    maxRapidity = cms.double(2.5)
    )

recoJets   =cms.Sequence(fixedGridRhoFastjetAllCalo+
                         fixedGridRhoFastjetCentralCalo+
                         ak4CaloJets
                         )

recoAllJets=cms.Sequence(fixedGridRhoFastjetAllCalo+
                         fixedGridRhoFastjetCentralCalo+                         
                         fixedGridRhoFastjetAllCalo+
                         ak4CaloJets
			 )


recoAllJetsPUOffsetCorr=cms.Sequence(fixedGridRhoFastjetAllCalo+
                                     fixedGridRhoFastjetCentralCalo+
                                     ak4CaloJetsPUCorr)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018

from RecoHI.HiJetAlgos.HiRecoJets_cff import caloTowersRec, caloTowers, akPu3CaloJets, akPu4CaloJets, akPu5CaloJets

recoJetsHI =cms.Sequence(fixedGridRhoFastjetAllCalo+
                         fixedGridRhoFastjetCentralCalo+
                         ak4CaloJets+
			 caloTowersRec+
			 caloTowers+
			 akPu3CaloJets+
			 akPu4CaloJets+
			 akPu5CaloJets
                         )

for e in [pp_on_AA_2018]:
 e.toReplaceWith(recoJets, recoJetsHI)
