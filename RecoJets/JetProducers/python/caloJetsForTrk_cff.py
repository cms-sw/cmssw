import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4CaloJets_cfi import ak4CaloJets as _ak4CaloJets
from RecoHI.HiJetAlgos.HiRecoJets_cff import akPu4CaloJets as _akPu4CaloJets
from RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi import calotowermaker
caloTowerForTrk = calotowermaker.clone(hbheInput=cms.InputTag('hbheprereco'))
ak4CaloJetsForTrk = _ak4CaloJets.clone(srcPVs = cms.InputTag('firstStepPrimaryVerticesUnsorted'), src= cms.InputTag('caloTowerForTrk'))
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toReplaceWith(ak4CaloJetsForTrk, _akPu4CaloJets.clone(srcPVs = cms.InputTag('firstStepPrimaryVerticesUnsorted'), src= cms.InputTag('caloTowerForTrk')))
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(ak4CaloJetsForTrk,
    srcPVs = "pixelVertices"
)

caloJetsForTrkTask = cms.Task(caloTowerForTrk,ak4CaloJetsForTrk)
caloJetsForTrk = cms.Sequence(caloJetsForTrkTask)

from Configuration.Eras.Modifier_pf_badHcalMitigation_cff import pf_badHcalMitigation
pf_badHcalMitigation.toModify( caloTowerForTrk, missingHcalRescaleFactorForEcal = 1.0 )

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toModify( caloTowerForTrk, hbheInput = cms.InputTag("hbhereco") )

