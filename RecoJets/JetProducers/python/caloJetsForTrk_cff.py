import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4CaloJets_cfi import ak4CaloJets
from RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi import calotowermaker
caloTowerForTrk = calotowermaker.clone(hbheInput=cms.InputTag('hbheprereco'))
ak4CaloJetsForTrk = ak4CaloJets.clone(srcPVs = cms.InputTag('firstStepPrimaryVertices'), src= cms.InputTag('caloTowerForTrk'))
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(ak4CaloJetsForTrk,
    srcPVs = "pixelVertices"
)
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
trackingPhase1PU70.toModify(ak4CaloJetsForTrk,
    srcPVs = "pixelVertices"
)
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(ak4CaloJetsForTrk,
    srcPVs = "pixelVertices"
)

caloJetsForTrk = cms.Sequence(caloTowerForTrk*ak4CaloJetsForTrk)

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toModify( caloTowerForTrk, hbheInput = cms.InputTag("hbhereco") )

