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

from RecoHI.HiJetAlgos.HiRecoJets_cff import caloTowersRec, caloTowers, akPu4CaloJets

recoJetsHI =cms.Sequence(fixedGridRhoFastjetAllCalo+
                         fixedGridRhoFastjetCentralCalo+
                         ak4CaloJets+
                         caloTowersRec+
                         caloTowers+
                         akPu4CaloJets
                         )
