import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.Configuration.ecalLocalReco_EventContentCosmics_cff import *
#
# start with HCAL part
#
#FEVT
RecoLocalCaloFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hbhereco_*_*',
                                           'keep *_hbheprereco_*_*', 
                                           'keep *_hfreco_*_*', 
                                           'keep *_horeco_*_*',
                                           'keep HBHERecHitsSorted_hbherecoMB_*_*',
                                           'keep HBHERecHitsSorted_hbheprerecoMB_*_*',
                                           'keep HORecHitsSorted_horecoMB_*_*',
                                           'keep HFRecHitsSorted_hfrecoMB_*_*',
                                           'keep ZDCDataFramesSorted_*Digis_*_*',
                                           'keep ZDCRecHitsSorted_*_*_*',
                                           'keep HcalUnpackerReport_*_*_*'
        )
)
#RECO content
RecoLocalCaloRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hbhereco_*_*', 
                                           'keep *_hfreco_*_*', 
                                           'keep *_horeco_*_*',
                                           'keep HBHERecHitsSorted_hbherecoMB_*_*',
                                           'keep HORecHitsSorted_horecoMB_*_*',
                                           'keep HFRecHitsSorted_hfrecoMB_*_*',
                                           #'keep ZDCDataFramesSorted_*Digis_*_*',
                                           'keep ZDCDataFramesSorted_hcalDigis_*_*',
                                           'keep ZDCDataFramesSorted_simHcalUnsuppressedDigis_*_*',
                                           'keep ZDCRecHitsSorted_*_*_*',
                                           #'keep HcalUnpackerReport_*_*_*'
                                           'keep HcalUnpackerReport_castorDigis_*_*',
                                           'keep HcalUnpackerReport_hcalDigis_*_*'
        )
)
#AOD content
RecoLocalCaloAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoLocalCaloFEVT.outputCommands.extend(ecalLocalRecoFEVT.outputCommands)
RecoLocalCaloRECO.outputCommands.extend(ecalLocalRecoRECO.outputCommands)
RecoLocalCaloAOD.outputCommands.extend(ecalLocalRecoAOD.outputCommands)

