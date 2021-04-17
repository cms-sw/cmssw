import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4CaloJets_cfi import ak4CaloJets as _ak4CaloJets
from RecoHI.HiJetAlgos.HiRecoJets_cff import akPu4CaloJets as _akPu4CaloJets
from RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi import calotowermaker
caloTowerForTrk = calotowermaker.clone(
    hbheInput='hbheprereco'
)

ak4CaloJetsForTrk = _ak4CaloJets.clone(
    srcPVs = 'firstStepPrimaryVerticesUnsorted', 
    src    = 'caloTowerForTrk'
)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toReplaceWith(ak4CaloJetsForTrk, _akPu4CaloJets.clone(
    srcPVs = 'firstStepPrimaryVerticesUnsorted', 
    src    = 'caloTowerForTrk')
)

from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(ak4CaloJetsForTrk,
    srcPVs = "pixelVertices"
)

caloJetsForTrkTask = cms.Task(caloTowerForTrk,ak4CaloJetsForTrk)
caloJetsForTrk = cms.Sequence(caloJetsForTrkTask)

from Configuration.Eras.Modifier_pf_badHcalMitigation_cff import pf_badHcalMitigation
pf_badHcalMitigation.toModify( caloTowerForTrk, missingHcalRescaleFactorForEcal = 1.0 )

from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toModify( caloTowerForTrk, hbheInput = "hbhereco" )
