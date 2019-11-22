import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4CaloJets_cfi import ak4CaloJets

from RecoJets.JetProducers.fixedGridRhoProducerFastjet_cfi import fixedGridRhoFastjetAllCalo

doPileup = cms.bool(True)

ak4CaloJetsPUCorr           =ak4CaloJets.clone           (doPUOffsetCorr = doPileup)
ak4CaloJets.doAreaFastjet = True

fixedGridRhoFastjetCentralCalo = fixedGridRhoFastjetAllCalo.clone(
    maxRapidity = cms.double(2.5)
    )

recoJetsTask   =cms.Task(fixedGridRhoFastjetAllCalo,
                         fixedGridRhoFastjetCentralCalo,
                         ak4CaloJets)
recoJets   =cms.Sequence(recoJetsTask)

recoAllJetsTask=cms.Task(fixedGridRhoFastjetAllCalo,
                         fixedGridRhoFastjetCentralCalo,                         
                         fixedGridRhoFastjetAllCalo,
                         ak4CaloJets)
recoAllJets=cms.Sequence(recoAllJetsTask)

recoAllJetsPUOffsetCorrTask=cms.Task(fixedGridRhoFastjetAllCalo,
                                     fixedGridRhoFastjetCentralCalo,
                                     ak4CaloJetsPUCorr)
recoAllJetsPUOffsetCorr=cms.Sequence(recoAllJetsPUOffsetCorrTask)

from RecoHI.HiJetAlgos.HiRecoJets_cff import caloTowersRecTask, caloTowers, akPu4CaloJets

recoJetsHITask =cms.Task(fixedGridRhoFastjetAllCalo,
                         fixedGridRhoFastjetCentralCalo,
                         ak4CaloJets,
                         caloTowersRecTask,
                         caloTowers,
                         akPu4CaloJets
                         )
recoJetsHI =cms.Sequence(recoJetsHITask)
