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

#Full Event content 
FastSimCoreFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_famosPileUp_*_*', 
        'keep *_famosSimHits_*_*', 
        'keep edmHepMCProduct_source_*_*', 
        'keep edmGenInfoProduct_source_*_*', 
        'keep *_genParticles_*_*', 
        'keep *_genEventWeight_*_*', 
        'keep *_genEventScale_*_*', 
        'keep *_genEventProcID_*_*', 
        'keep *_genEventRunInfo_*_*', 
        'keep edmAlpgenInfoProduct_source_*_*')
)

#RECO content
FastSimCoreRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_genParticles_*_*', 
        'keep *_genEventWeight_*_*', 
        'keep *_genEventScale_*_*', 
        'keep edmHepMCProduct_source_*_*', 
        'keep edmGenInfoProduct_source_*_*', 
        'keep SimTracks_famosSimHits_*_*', 
        'keep SimVertexs_famosSimHits_*_*', 
        'keep *_genEventProcID_*_*', 
        'keep *_genEventRunInfo_*_*', 
        'keep edmAlpgenInfoProduct_source_*_*')
)

#AOD content
FastSimCoreAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmGenInfoProduct_source_*_*', 
        'keep *_genParticles_*_*', 
        'keep *_genEventWeight_*_*', 
        'keep *_genEventScale_*_*', 
        'keep *_genEventProcID_*_*', 
        'keep *_genEventRunInfo_*_*', 
        'keep edmAlpgenInfoProduct_source_*_*')
)

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
# The Calo RecHits are also different
#####################################################################

#Full Event content 
FastSimRecoLocalCaloFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_caloRecHits_*_*', 
        'keep *_hcalRecHits_*_*')
)

#RECO content
FastSimRecoLocalCaloRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_caloRecHits_*_*', 
        'keep *_hcalRecHits_*_*')
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
    outputCommands = cms.untracked.vstring('keep *_iterativeGSWithMaterialTracks_*_*')
)

#RECO content
FastSimRecoTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_iterativeGSWithMaterialTracks_*_*')
)

#AOD content
FastSimRecoTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_iterativeGSWithMaterialTracks_*_*')
)

#####################################################################
# The Muons are, for now, parameterized
#####################################################################

#Full Event content 
FastSimParamMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_paramMuons_*_*', 
        'keep l1extraL1MuonParticles_l1ParamMuons_*_*')
)

#MuIsoDeposits
FastSimParamMuonFEVT.outputCommands.extend(RecoMuonIsolationParamGlobal.outputCommands)

#RECO content
FastSimParamMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_paramMuons_*_*', 
        'keep l1extraL1MuonParticles_l1ParamMuons_*_*')
)

#MuIsoDeposits
FastSimParamMuonRECO.outputCommands.extend(RecoMuonIsolationParamGlobal.outputCommands)

#AOD content
FastSimParamMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoMuons_paramMuons_*_*', 
        'keep *_muParamGlobalIsoDepositCtfTk_*_*', 
        'keep *_muParamGlobalIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muParamGlobalIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muParamGlobalIsoDepositJets_*_*', 
        'keep l1extraL1MuonParticles_l1ParamMuons_*_*')
)

# Addition to the event content
# We don't need to remove anything, as the corresponding products are
# not produced anyway in a FastSimulation job.
#####################################################################
#
# FEVT Data Tier re-definition
#
#####################################################################

FEVTEventContent.outputCommands.extend(FastSimRecoLocalTrackerFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(FastSimRecoLocalCaloFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(FastSimRecoTrackerFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(FastSimParamMuonFEVT.outputCommands)

#####################################################################
#
# RECO Data Tier definition
#
#####################################################################

RECOEventContent.outputCommands.extend(FastSimRecoLocalTrackerRECO.outputCommands)
RECOEventContent.outputCommands.extend(FastSimRecoLocalCaloRECO.outputCommands)
RECOEventContent.outputCommands.extend(FastSimRecoTrackerRECO.outputCommands)
RECOEventContent.outputCommands.extend(FastSimParamMuonRECO.outputCommands)

#####################################################################
#
# AOD Data Tier definition
#
#####################################################################

AODEventContent.outputCommands.extend(FastSimRecoLocalTrackerAOD.outputCommands)
AODEventContent.outputCommands.extend(FastSimRecoLocalCaloAOD.outputCommands)
AODEventContent.outputCommands.extend(FastSimRecoTrackerAOD.outputCommands)
AODEventContent.outputCommands.extend(FastSimParamMuonAOD.outputCommands)

####################################################################
#
# FEVTSIM Data Tier re-definition
#
#####################################################################

FEVTSIMEventContent.outputCommands.extend(FastSimCoreFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(FastSimRecoLocalTrackerFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(FastSimRecoLocalCaloFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(FastSimRecoTrackerFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(FastSimParamMuonFEVT.outputCommands)

#####################################################################
#
# FEVTSIMDIGI Data Tier re-definition
#
#####################################################################

FEVTSIMDIGIEventContent.outputCommands.extend(FastSimCoreFEVT.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(FastSimRecoLocalTrackerFEVT.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(FastSimRecoLocalCaloFEVT.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(FastSimRecoTrackerFEVT.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(FastSimParamMuonFEVT.outputCommands)

#####################################################################
#
# RECOSIM Data Tier re-definition
#
#####################################################################

RECOSIMEventContent.outputCommands.extend(FastSimCoreRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(FastSimRecoLocalTrackerRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(FastSimRecoLocalCaloRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(FastSimRecoTrackerRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(FastSimParamMuonRECO.outputCommands)

#####################################################################
#
# AODSIM Data Tier re-definition
#
#####################################################################

AODSIMEventContent.outputCommands.extend(FastSimCoreAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(FastSimRecoLocalTrackerAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(FastSimRecoLocalCaloAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(FastSimRecoTrackerAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(FastSimParamMuonAOD.outputCommands)

#####################################################################
#
# FEVTSIMHLTDEBUG Data Tier re-definition
#
#####################################################################

FEVTSIMHLTDEBUGEventContent.outputCommands.extend(FEVTSIMEventContent.outputCommands)

#####################################################################
#
# FEVTSIMDIGIHLTDEBUG  Data Tier re-definition
#
#####################################################################
FEVTSIMDIGIHLTDEBUGEventContent.outputCommands.extend(FEVTSIMDIGIEventContent.outputCommands)


