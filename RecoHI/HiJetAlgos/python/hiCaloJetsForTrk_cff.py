import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi import calotowermaker
from RecoHI.HiJetAlgos.HiRecoJets_cff import akPu4CaloJets
from JetMETCorrections.Configuration.DefaultJEC_cff import *


hiCaloTowerForTrk  = calotowermaker.clone( hbheInput = 'hbheprereco')
akPu4CaloJetsForTrk = akPu4CaloJets.clone( srcPVs = 'hiSelectedPixelVertex', src = 'hiCaloTowerForTrk')

akPu4CaloJetsCorrected  = ak4CaloJetsL2L3.clone(
    src = "akPu4CaloJetsForTrk"
)

akPu4CaloJetsSelected = cms.EDFilter( "LargestEtCaloJetSelector",
    src = cms.InputTag( "akPu4CaloJetsCorrected" ),
    filter = cms.bool( False ),
    maxNumber = cms.uint32( 4 )
)

hiCaloJetsForTrkTask = cms.Task(hiCaloTowerForTrk,akPu4CaloJetsForTrk,akPu4CaloJetsCorrected,akPu4CaloJetsSelected)
