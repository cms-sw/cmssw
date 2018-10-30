import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.Configuration.ecalLocalReco_EventContent_cff import *
#
# start with HCAL part
#
#FEVT
RecoLocalCaloFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hbhereco_*_*',
                                           'keep *_hbheprereco_*_*',
                                           'keep *_hfprereco_*_*',
                                           'keep *_hfreco_*_*',
                                           'keep *_horeco_*_*',
                                           'keep HBHERecHitsSorted_hbherecoMB_*_*',
                                           'keep HBHERecHitsSorted_hbheprerecoMB_*_*',
                                           'keep HORecHitsSorted_horecoMB_*_*',
                                           'keep HFRecHitsSorted_hfrecoMB_*_*',
                                           'keep ZDCDataFramesSorted_*Digis_*_*',
                                           'keep ZDCRecHitsSorted_*_*_*',
                                           'keep QIE10DataFrameHcalDataFrameContainer_hcalDigis_ZDC_*',
                                           'keep *_reducedHcalRecHits_*_*',
                                           'keep *_castorreco_*_*',
                                           'keep HcalUnpackerReport_*_*_*'
        )
)
#RECO content
RecoLocalCaloRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hbhereco_*_*',
                                           'keep *_hbheprereco_*_*',
                                           'keep *_hfprereco_*_*',
                                           'keep *_hfreco_*_*',
                                           'keep *_horeco_*_*',
                                           'keep HBHERecHitsSorted_hbherecoMB_*_*',
                                           'keep HORecHitsSorted_horecoMB_*_*',
                                           'keep HFRecHitsSorted_hfrecoMB_*_*',
                                           #'keep ZDCDataFramesSorted_*Digis_*_*',
                                           'keep ZDCDataFramesSorted_hcalDigis_*_*',
                                           'keep ZDCDataFramesSorted_castorDigis_*_*',
                                           'keep QIE10DataFrameHcalDataFrameContainer_hcalDigis_ZDC_*',
                                           'keep ZDCRecHitsSorted_zdcreco_*_*',
                                           'keep *_reducedHcalRecHits_*_*',
                                           'keep *_castorreco_*_*',
                                           #'keep HcalUnpackerReport_*_*_*'
                                           'keep HcalUnpackerReport_castorDigis_*_*',
                                           'keep HcalUnpackerReport_hcalDigiAlCaMB_*_*',
                                           'keep HcalUnpackerReport_hcalDigis_*_*'
        )
)
#AOD content
RecoLocalCaloAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep *_castorreco_*_*',
    'keep *_reducedHcalRecHits_*_*',
    #'keep HcalUnpackerReport_*_*_*'
    'keep HcalUnpackerReport_castorDigis_*_*',
    'keep HcalUnpackerReport_hcalDigiAlCaMB_*_*',
    'keep HcalUnpackerReport_hcalDigis_*_*'
    )
)
RecoLocalCaloFEVT.outputCommands.extend(ecalLocalRecoFEVT.outputCommands)
RecoLocalCaloRECO.outputCommands.extend(ecalLocalRecoRECO.outputCommands)
RecoLocalCaloAOD.outputCommands.extend(ecalLocalRecoAOD.outputCommands)

def _updateOutput( era, outputPSets, commands):
   for o in outputPSets:
      era.toModify( o, outputCommands = o.outputCommands + commands )

# mods for HGCAL
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( RecoLocalCaloFEVT, outputCommands = RecoLocalCaloFEVT.outputCommands + [
        'keep *_HGCalRecHit_*_*',
        'keep *_HGCalUncalibRecHit_*_*',
        'keep *_hgcalLayerClusters_*_*',
        'keep *_hgcalMultiClusters_*_*'
    ]
)
phase2_hgcal.toModify( RecoLocalCaloRECO, outputCommands = RecoLocalCaloRECO.outputCommands + ['keep *_HGCalRecHit_*_*','keep *_hgcalLayerClusters_*_*', 'keep *_hgcalMultiClusters_*_*'] )
# don't modify AOD for HGCal yet, need "reduced" rechits collection first (i.e. requires reconstruction)
phase2_hgcal.toModify( RecoLocalCaloAOD, outputCommands = RecoLocalCaloAOD.outputCommands + ['keep *_HGCalRecHit_*_*','keep *_hgcalLayerClusters_*_*'] )

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
(pA_2016|pp_on_AA_2018).toModify( RecoLocalCaloAOD.outputCommands,
                  func=lambda outputCommands: outputCommands.extend(['keep *_zdcreco_*_*',
                                                                     'keep ZDCDataFramesSorted_hcalDigis_*_*',
                                                                     'keep ZDCDataFramesSorted_castorDigis_*_*',
                                                                     'keep QIE10DataFrameHcalDataFrameContainer_hcalDigis_ZDC_*'
                                                                     ])
                  )
