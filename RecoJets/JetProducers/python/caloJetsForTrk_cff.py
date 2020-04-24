import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4CaloJets_cfi import ak4CaloJets as _ak4CaloJets
from RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi import calotowermaker
caloTowerForTrk = calotowermaker.clone(hbheInput=cms.InputTag('hbheprereco'))
ak4CaloJetsForTrk = _ak4CaloJets.clone(srcPVs = cms.InputTag('firstStepPrimaryVerticesUnsorted'), src= cms.InputTag('caloTowerForTrk'))
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(ak4CaloJetsForTrk,
    srcPVs = "pixelVertices"
)

caloJetsForTrk = cms.Sequence(caloTowerForTrk*ak4CaloJetsForTrk)

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toModify( caloTowerForTrk, hbheInput = cms.InputTag("hbhereco") )

