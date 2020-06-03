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
RAWEventContent.outputCommands.extend(RecoHIRAW.outputCommands)

#RECO
RECOEventContent.outputCommands.extend(RecoHIRECO.outputCommands)

#AOD
AODEventContent.outputCommands.extend(RecoHIAOD.outputCommands)

#RAWSIM
RAWSIMEventContent.outputCommands.extend(RecoHIRAW.outputCommands)
RAWSIMEventContent.outputCommands.extend(HiMixRAW.outputCommands)

#RAWSIMHLT
RAWSIMHLTEventContent.outputCommands.extend(RecoHIRAW.outputCommands)
RAWSIMHLTEventContent.outputCommands.extend(HiMixRAW.outputCommands)

#RECOSIM
RECOSIMEventContent.outputCommands.extend(RecoHIRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(HiMixRECO.outputCommands)

#AODSIM
AODSIMEventContent.outputCommands.extend(RecoHIAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(HiMixAOD.outputCommands)

#FEVT (RAW + RECO)
FEVTEventContent.outputCommands.extend(RecoHIFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoHIRAW.outputCommands)

#FEVTHLTALL (FEVT + all HLT)
FEVTHLTALLEventContent.outputCommands.extend(RecoHIFEVT.outputCommands)
FEVTHLTALLEventContent.outputCommands.extend(RecoHIRAW.outputCommands)

#FEVTSIM (RAWSIM + RECOSIM)
FEVTSIMEventContent.outputCommands.extend(RecoHIFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoHIRAW.outputCommands)
FEVTSIMEventContent.outputCommands.extend(HiMixRAW.outputCommands)

#RAW DEBUG(e.g. mergedtruth from trackingParticles) 
RAWDEBUGEventContent.outputCommands.extend(RecoHIRAW.outputCommands)
RAWDEBUGEventContent.outputCommands.extend(HiMixRAW.outputCommands)

#RAW HLT DEBUG 
RAWDEBUGHLTEventContent.outputCommands.extend(RecoHIRAW.outputCommands)
RAWDEBUGHLTEventContent.outputCommands.extend(HiMixRAW.outputCommands)

#RECO DEBUG  
RECODEBUGEventContent.outputCommands.extend(RecoHIRECO.outputCommands)
RECODEBUGEventContent.outputCommands.extend(HiMixRAW.outputCommands)

#FEVT DEBUG 
FEVTDEBUGEventContent.outputCommands.extend(RecoHIFEVT.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(RecoHIRAW.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(HiMixRAW.outputCommands)

#FEVT HLT DEBUG  
FEVTDEBUGHLTEventContent.outputCommands.extend(RecoHIFEVT.outputCommands)
FEVTDEBUGHLTEventContent.outputCommands.extend(RecoHIRAW.outputCommands)
FEVTDEBUGHLTEventContent.outputCommands.extend(HiMixRAW.outputCommands)
