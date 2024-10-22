import FWCore.ParameterSet.Config as cms

# Unprescale HLT_MET and HLT_SinglePhoton triggers
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltMonopole = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltMonopole.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
hltMonopole.HLTPaths = cms.vstring(
    #2016
    #"HLT_Photon175_v*",
    #"HLT_DoublePhoton60_v*",
    #"HLT_PFMET300_v*",
    #"HLT_PFMET170_HBHE_BeamHaloCleaned_v*",
    #2017,2018
    #"HLT_Photon200_v*",
    #"HLT_Photon300_NoHE_v*",
    #"HLT_DoublePhoton70_v*",
    #"HLT_PFMET140_PFMHT140_IDTight_v*",
    #"HLT_PFMET250_HBHECleaned_v*",
    #"HLT_PFMET300_HBHECleaned_v*",
    #2021: on EGamma dataset only
    #"HLT_Photon200_v*",
    #"HLT_DoublePhoton70_v*",
    #"HLT_DoublePhoton85_v*",
    #2024: on EGamma, MET and JetMET datasets
    "HLT_Photon200_v*",
    "HLT_PFMET200_BeamHaloCleaned_v*"
)
hltMonopole.throw = False
hltMonopole.andOr = True

from Configuration.EventContent.EventContent_cff import AODEventContent
EXOMonopoleSkimContent = AODEventContent.clone()
EXOMonopoleSkimContent.outputCommands.append('drop *')
EXOMonopoleSkimContent.outputCommands.append('keep *_hybridSuperClusters_*_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_multi5x5SuperClusters_multi5x5EndcapBasicClusters_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_multi5x5SuperClusters_multi5x5EndcapSuperClusters_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_multi5x5SuperClusters_uncleanOnlyMulti5x5EndcapBasicClusters_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_multi5x5SuperClusters_uncleanOnlyMulti5x5EndcapSuperClusters_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_uncleanSCRecovered_uncleanHybridBarrelBasicClusters_*') 
EXOMonopoleSkimContent.outputCommands.append('keep *_uncleanEERecovered_uncleanEndcapBasicClusters_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_siStripClusters_*_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_siPixelClusters_*_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_generalTracks_*_*')
EXOMonopoleSkimContent.outputCommands.append('drop *_generalTracks_QualityMasks_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_ecalRecHit_EcalRecHitsEB_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_ecalRecHit_EcalRecHitsEE_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_hbhereco_*_*')
EXOMonopoleSkimContent.outputCommands.append('keep edmTriggerResults_TriggerResults_*_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_hltTriggerSummaryAOD_*_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_offlinePrimaryVertices_*_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_gedGsfElectrons_*_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_photons_*_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_pfMet_*_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_offlineBeamSpot_*_*')
EXOMonopoleSkimContent.outputCommands.append('keep *_siPixelDigis_*_*')

# monopole skim sequence
EXOMonopoleSkimSequence = cms.Sequence(hltMonopole)
