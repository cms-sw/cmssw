import FWCore.ParameterSet.Config as cms

#  Heavy Ions Event Content including:
#    1) common event content from standard file (EventContent_cff)
#    2) heavy-ion specific content from other subsystems (e.g. HiMixing)
#    3) heavy-ion specific reconstruction content from RecoHI

# Common Subsystems
from Configuration.EventContent.EventContent_cff import *

# Heavy-Ion Specific Event Content
from SimGeneral.Configuration.SimGeneral_HiMixing_EventContent_cff import * # heavy ion signal mixing
from RecoHI.Configuration.RecoHI_EventContent_cff import *       # heavy ion reconstruction


#RAW
RAWEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
	'keep FEDRawDataCollection_rawDataRepacker_*_*',
        'keep FEDRawDataCollection_hybridRawDataRepacker_*_*',
        'keep FEDRawDataCollection_virginRawDataRepacker_*_*')
)

#RECO
RECOEventContent.outputCommands.extend(RecoHIRECO.outputCommands)

#AOD
AODEventContent.outputCommands.extend(RecoHIAOD.outputCommands)

#RAWSIM
RAWSIMEventContent.outputCommands.extend(RAWEventContent.outputCommands)
RAWSIMEventContent.outputCommands.extend(HiMixRAW.outputCommands)

#RAWSIMHLT
RAWSIMHLTEventContent.outputCommands.extend(RAWSIMEventContent.outputCommands)

#RECOSIM
RECOSIMEventContent.outputCommands.extend(RecoHIRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(HiMixRECO.outputCommands)

#AODSIM
AODSIMEventContent.outputCommands.extend(RecoHIAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(HiMixAOD.outputCommands)

#FEVT (RAW + RECO)
FEVTEventContent.outputCommands.extend(RecoHIFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RAWEventContent.outputCommands)

#FEVTHLTALL (FEVT + all HLT)
FEVTHLTALLEventContent.outputCommands.extend(FEVTEventContent.outputCommands)

#FEVTSIM (RAWSIM + RECOSIM)
FEVTSIMEventContent.outputCommands.extend(FEVTEventContent.outputCommands)
FEVTSIMEventContent.outputCommands.extend(HiMixRAW.outputCommands)

#RAW DEBUG(e.g. mergedtruth from trackingParticles) 
RAWDEBUGEventContent.outputCommands.extend(RAWEventContent.outputCommands)
RAWDEBUGEventContent.outputCommands.extend(HiMixRAW.outputCommands)

#RAW HLT DEBUG 
RAWDEBUGHLTEventContent.outputCommands.extend(RAWDEBUGEventContent.outputCommands)

#RECO DEBUG  
RECODEBUGEventContent.outputCommands.extend(RecoHIRECO.outputCommands)
RECODEBUGEventContent.outputCommands.extend(HiMixRAW.outputCommands)

#FEVT DEBUG 
FEVTDEBUGEventContent.outputCommands.extend(FEVTSIMEventContent.outputCommands)

#FEVT HLT DEBUG  
FEVTDEBUGHLTEventContent.outputCommands.extend(FEVTSIMEventContent.outputCommands)
