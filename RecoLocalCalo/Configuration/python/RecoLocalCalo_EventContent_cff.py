import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.Configuration.ecalLocalReco_EventContent_cff import *
#
# start with HCAL part
#

#AOD content
RecoLocalCaloAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep *_castorreco_*_*',
    'keep *_reducedHcalRecHits_*_*',
    'keep HcalUnpackerReport_castorDigis_*_*',
    'keep HcalUnpackerReport_hcalDigiAlCaMB_*_*',
    'keep HcalUnpackerReport_hcalDigis_*_*')
)
RecoLocalCaloAOD.outputCommands.extend(ecalLocalRecoAOD.outputCommands)
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose
from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from Configuration.ProcessModifiers.storeZDCDigis_cff import storeZDCDigis
# don't modify AOD for HGCal yet, need "reduced" rechits collection first (i.e. requires reconstruction)
phase2_hgcal.toModify( RecoLocalCaloAOD, 
    outputCommands = RecoLocalCaloAOD.outputCommands + ['keep *_HGCalRecHit_*_*',
                                                        'keep recoCaloClusters_hgcalMergeLayerClusters_*_*',
                                                        'keep *_hgcalMergeLayerClusters_timeLayerCluster_*',
                                                        'keep *_hgcalMergeLayerClusters_InitialLayerClustersMask_*'])
phase2_hfnose.toModify( RecoLocalCaloAOD, 
    outputCommands = RecoLocalCaloAOD.outputCommands + ['keep recoCaloClusters_hgcalLayerClustersHFNose_*_*',
                                                        'keep *_hgcalLayerClustersHFNose_timeLayerCluster_*',
                                                        'keep *_hgcalLayerClustersHFNose_InitialLayerClustersMask_*'])
(pA_2016|pp_on_AA).toModify( RecoLocalCaloAOD.outputCommands,
        func=lambda outputCommands: outputCommands.extend(['keep *_zdcreco_*_*',
                                                           'keep ZDCDataFramesSorted_hcalDigis_*_*',
                                                           'keep ZDCDataFramesSorted_castorDigis_*_*',
                                                           'keep QIE10DataFrameHcalDataFrameContainer_hcalDigis_ZDC_*'])
        )
storeZDCDigis.toModify( RecoLocalCaloAOD,
                        outputCommands = RecoLocalCaloAOD.outputCommands + ['keep QIE10DataFrameHcalDataFrameContainer_hcalDigis_ZDC_*'])
from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify( RecoLocalCaloAOD, 
    outputCommands = RecoLocalCaloAOD.outputCommands + ['keep *_towerMaker_*_*',
                                                        'keep *_zdcreco_*_*',
                                                        'keep ZDCDataFramesSorted_hcalDigis_*_*',
                                                        'keep ZDCDataFramesSorted_castorDigis_*_*',
                                                        'keep QIE10DataFrameHcalDataFrameContainer_hcalDigis_ZDC_*'])

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
                                           'keep ZDCDataFramesSorted_hcalDigis_*_*',
                                           'keep ZDCDataFramesSorted_castorDigis_*_*',
                                           'keep QIE10DataFrameHcalDataFrameContainer_hcalDigis_ZDC_*',
                                           'keep ZDCRecHitsSorted_zdcreco_*_*')
)
RecoLocalCaloRECO.outputCommands.extend(RecoLocalCaloAOD.outputCommands)
RecoLocalCaloRECO.outputCommands.extend(ecalLocalRecoRECO.outputCommands)
phase2_hgcal.toModify( RecoLocalCaloRECO, 
    outputCommands = RecoLocalCaloRECO.outputCommands + ['keep *_hgcalMultiClusters_*_*',
                                                         'keep *_iterHGCalMultiClusters_*_*'])

#FEVT content
RecoLocalCaloFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep HBHERecHitsSorted_hbheprerecoMB_*_*',
                                           'keep ZDCDataFramesSorted_*Digis_*_*',
                                           'keep ZDCRecHitsSorted_*_*_*',
                                           'keep HcalUnpackerReport_*_*_*')
)
RecoLocalCaloFEVT.outputCommands.extend(RecoLocalCaloRECO.outputCommands)
RecoLocalCaloFEVT.outputCommands.extend(ecalLocalRecoFEVT.outputCommands)
phase2_hgcal.toModify( RecoLocalCaloFEVT, 
    outputCommands = RecoLocalCaloFEVT.outputCommands + ['keep *_HGCalUncalibRecHit_*_*'])
