import FWCore.ParameterSet.Config as cms

# HLT dimuon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltOniaUPCHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltOniaUPCHI.HLTPaths = ["HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrack_v*"]
hltOniaUPCHI.throw = False
hltOniaUPCHI.andOr = True

# UPC double mu skim sequence
oniaUPCSkimSequence = cms.Sequence(
    hltOniaUPCHI 
    )
