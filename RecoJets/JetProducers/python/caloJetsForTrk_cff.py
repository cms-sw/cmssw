import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

from RecoJets.JetProducers.ak4CaloJets_cfi import ak4CaloJets
from RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi import calotowermaker
caloTowerForTrk = calotowermaker.clone(hbheInput=cms.InputTag('hbheprereco'))
ak4CaloJetsForTrk = ak4CaloJets.clone(srcPVs = cms.InputTag('firstStepPrimaryVertices'), src= cms.InputTag('caloTowerForTrk'))
eras.trackingLowPU.toModify(ak4CaloJetsForTrk,
    srcPVs = "pixelVertices"
)
eras.trackingPhase1PU70.toModify(ak4CaloJetsForTrk,
    srcPVs = "pixelVertices"
)

caloJetsForTrk = cms.Sequence(caloTowerForTrk*ak4CaloJetsForTrk)

from Configuration.StandardSequences.Eras import eras
eras.phase2_hcal.toModify( caloTowerForTrk, hbheInput = cms.InputTag("hbheUpgradeReco"), hfInput = cms.InputTag("hfUpgradeReco") )
