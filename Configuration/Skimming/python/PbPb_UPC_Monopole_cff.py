import FWCore.ParameterSet.Config as cms

# HLT UPC pixel thrust trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltMonopole = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltMonopole.HLTPaths = ["HLT_HIUPC_MinPixelThrust0p8_MaxPixelCluster10000_v*"]
hltMonopole.throw = False
hltMonopole.andOr = True

from Configuration.Skimming.PDWG_EXOMONOPOLE_cff import EXOMonopoleSkimContent
upcMonopoleSkimContent = EXOMonopoleSkimContent.clone()
upcMonopoleSkimContent.outputCommands.append('keep FEDRawDataCollection_rawDataRepacker_*_*')

# UPC monopole skim sequence
upcMonopoleSkimSequence = cms.Sequence(hltMonopole)
