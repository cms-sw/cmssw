import FWCore.ParameterSet.Config as cms
#####################################################################
#
# Event Content definition
#
#####################################################################
from Configuration.EventContent.EventContent_cff import *

#####################################################################
# The simHits part is definitely different in FastSim
#####################################################################



FastSimCoreFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_famosPileUp_*_*', 
        'keep *_famosSimHits_*_*',
        'drop *_g4SimHits_*_*', # if an EDAlias creates this, it must be dropped to avoid a clash
        'keep *_MuonSimHits_*_*')
)

FastSimCoreFEVT.outputCommands.extend(GeneratorInterfaceRAW.outputCommands)


#RECO content
FastSimCoreRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep SimTracks_famosSimHits_*_*', 
        'keep SimVertexs_famosSimHits_*_*')
)
FastSimCoreRECO.outputCommands.extend(GeneratorInterfaceRECO.outputCommands)

#AOD content
FastSimCoreAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
FastSimCoreAOD.outputCommands.extend(GeneratorInterfaceAOD.outputCommands)

#####################################################################
# The Tracker RecHits are also different
#####################################################################

#Full Event content 
FastSimRecoLocalTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_siTrackerGaussianSmearingRecHits_*_*')
)
#RECO content
FastSimRecoLocalTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
#AOD content
FastSimRecoLocalTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)


#####################################################################
# CaloJet+Tracks are apparently saved nowhere
# Let's save them in the fast simulation (AOD only)
#####################################################################
FastSimCJPT = cms.PSet(
    outputCommands = cms.untracked.vstring(
         'keep *_JetPlusTrackZSPCorJetIcone5_*_*',
         'keep *_ZSPJetCorJetIcone5_*_*'
    )
)

#####################################################################
# The Calo RecHits are also different
#####################################################################

#Full Event content 
FastSimRecoLocalCaloFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_caloRecHits_*_*', 
        'keep *_hcalRecHits_*_*',
        'keep EBDigiCollection_ecalRecHit_*_*',
        'keep EEDigiCollection_ecalRecHit_*_*')
#        'keep *_particleFlowRecHit_*_*')
)

#RECO content
FastSimRecoLocalCaloRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_caloRecHits_*_*', 
        'keep *_hcalRecHits_*_*',
        'keep EBDigiCollection_ecalRecHit_*_*',
        'keep EEDigiCollection_ecalRecHit_*_*')
#        'keep *_particleFlowRecHit_*_*')
)

#AOD content
FastSimRecoLocalCaloAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

#####################################################################
# The Tracker Tracks are also different
#####################################################################

#Full Event content 
FastSimRecoTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_iterativeGSWithMaterialTracks_*_*',
                                           'keep *_generalTracksBeforeMixing_*_*')
    )

#RECO content
FastSimRecoTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_iterativeGSWithMaterialTracks_*_*',
                                           'keep *_generalTracksBeforeMixing_*_*')
)

#AOD content
FastSimRecoTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_iterativeGSWithMaterialTracks_*_*',
                                           #'keep *_generalTracksBeforeMixing_*_*'
                                           )
)


#####################################################################
# new Particle Flow Collection with "Fake" Neutral Hadrons
#####################################################################

#Full Event content 
FastSimParticleFlowFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoPFCandidates_FSparticleFlow_*_*',
                                           #stuff added for two-step processing (simWithSomeReco followed by reconstructionHighLevel):
                                           'keep *_muon*_*_*',
                                           'keep *_towerMaker*_*_*',
                                           'keep *_particleFlow*_*_*',
                                           'keep *_pf*_*_*',
                                           'keep *_*DetId*_*_*')
)

#RECO content 
FastSimParticleFlowRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoPFCandidates_FSparticleFlow_*_*')
)

#AOD content 
FastSimParticleFlowAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoPFCandidates_FSparticleFlow_*_*')
)




# Addition to the event content
# We don't need to remove anything, as the corresponding products are
# not produced anyway in a FastSimulation job.

#####################################################################
#
# AOD Data Tier definition
#
#####################################################################

AODEventContent.outputCommands.extend(FastSimRecoLocalTrackerAOD.outputCommands)
AODEventContent.outputCommands.extend(FastSimRecoLocalCaloAOD.outputCommands)
AODEventContent.outputCommands.extend(FastSimRecoTrackerAOD.outputCommands)
AODEventContent.outputCommands.extend(FastSimParticleFlowAOD.outputCommands)

#####################################################################
#
# AODSIM Data Tier definition
#
#####################################################################

AODSIMEventContent.outputCommands.extend(FastSimCoreAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(FastSimRecoLocalTrackerAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(FastSimRecoLocalCaloAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(FastSimRecoTrackerAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(FastSimParticleFlowAOD.outputCommands)

#####################################################################
#
# RECO Data Tier definition
#
#####################################################################

RECOEventContent.outputCommands.extend(FastSimRecoLocalTrackerRECO.outputCommands)
RECOEventContent.outputCommands.extend(FastSimRecoLocalCaloRECO.outputCommands)
RECOEventContent.outputCommands.extend(FastSimRecoTrackerRECO.outputCommands)
RECOEventContent.outputCommands.extend(FastSimParticleFlowRECO.outputCommands)

#####################################################################
#
# RECOSIM Data Tier definition
#
#####################################################################

RECOSIMEventContent.outputCommands.extend(FastSimCoreRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(FastSimRecoLocalTrackerRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(FastSimRecoLocalCaloRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(FastSimRecoTrackerRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(FastSimParticleFlowRECO.outputCommands)

#####################################################################
#
# RECODEBUG Data Tier definition
#
#####################################################################

RECODEBUGEventContent.outputCommands.extend(FastSimCoreRECO.outputCommands)
RECODEBUGEventContent.outputCommands.extend(FastSimRecoLocalTrackerRECO.outputCommands)
RECODEBUGEventContent.outputCommands.extend(FastSimRecoLocalCaloRECO.outputCommands)
RECODEBUGEventContent.outputCommands.extend(FastSimRecoTrackerRECO.outputCommands)
RECODEBUGEventContent.outputCommands.extend(FastSimParticleFlowRECO.outputCommands)

####################################################################
#
# FEVT Data Tier re-definition
#
#####################################################################

FEVTEventContent.outputCommands.extend(FastSimRecoLocalTrackerFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(FastSimRecoLocalCaloFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(FastSimRecoTrackerFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(FastSimParticleFlowFEVT.outputCommands) 

####################################################################
#
# FEVTSIM Data Tier re-definition
#
#####################################################################

FEVTSIMEventContent.outputCommands.extend(FastSimCoreFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(FastSimRecoLocalTrackerFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(FastSimRecoLocalCaloFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(FastSimRecoTrackerFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(FastSimParticleFlowFEVT.outputCommands) 

#####################################################################
#
# FEVTDEBUG Data Tier re-definition
#
#####################################################################

FEVTDEBUGEventContent.outputCommands.extend(FastSimCoreFEVT.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(FastSimRecoLocalTrackerFEVT.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(FastSimRecoLocalCaloFEVT.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(FastSimRecoTrackerFEVT.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(FastSimParticleFlowFEVT.outputCommands) 

#####################################################################
#
# FEVTDEBUGHLT  Data Tier re-definition
#
#####################################################################
FEVTDEBUGHLTEventContent.outputCommands.extend(FastSimCoreFEVT.outputCommands)
FEVTDEBUGHLTEventContent.outputCommands.extend(FastSimRecoLocalTrackerFEVT.outputCommands)
FEVTDEBUGHLTEventContent.outputCommands.extend(FastSimRecoLocalCaloFEVT.outputCommands)
FEVTDEBUGHLTEventContent.outputCommands.extend(FastSimRecoTrackerFEVT.outputCommands)
FEVTDEBUGHLTEventContent.outputCommands.extend(FastSimParticleFlowFEVT.outputCommands) 

##################
# get rid of some edaliases in the output
##################
for _entry in [FEVTDEBUGEventContent,FEVTSIMEventContent,GENRAWEventContent,FEVTDEBUGHLTEventContent,HLTDEBUGEventContent,HLTDebugFEVT,HLTDebugRAW,RAWDEBUGHLTEventContent,RAWRECODEBUGHLTEventContent,RAWRECOSIMHLTEventContent,RAWSIMHLTEventContent,]:
    _entry.outputCommands.append('drop *_ecalPreshowerDigis_*_*')
    _entry.outputCommands.append('drop *_ecalDigis_*_*')
    _entry.outputCommands.append('drop *_hcalDigis_*_*')
    _entry.outputCommands.append('drop *_muonDTDigis_*_*')
    _entry.outputCommands.append('drop *_muonCSCDigis_*_*')
    _entry.outputCommands.append('drop *_muonRPCDigis_*_*')
    _entry.outputCommands.append('drop *_gtDigis_*_*')
    _entry.outputCommands.append('drop *_hltIter*_*_*')
    _entry.outputCommands.append('drop *_hlt*Digis_*_*')
    _entry.outputCommands.append('drop *_gmtDigis_*_*')

#####################################################################
#
# To be used only to create the MinBias sample for "new mixing" (--eventcontent=FASTPU)
#
#####################################################################

FASTPUEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
                                           'keep *_famosSimHits_*_*',
                                           'keep *_MuonSimHits_*_*',
                                           'drop *_famosSimHits_VertexTypes_*',    
                                           'keep *_generalTracksBeforeMixing_*_*',
                                           'drop *_generalTracksBeforeMixing_MVAValues_*',
                                           'drop *_generalTracksBeforeMixing_QualityMasks_*',
                                           'keep edmHepMCProduct_generatorSmeared_*_*'
                                           )
    )


PREMIXEventContent.outputCommands.extend(['keep *_mix_generalTracks_*'])
