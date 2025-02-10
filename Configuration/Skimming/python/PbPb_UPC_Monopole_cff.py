import FWCore.ParameterSet.Config as cms

# HLT UPC pixel thrust trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltUPCMonopole = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltUPCMonopole.HLTPaths = ["HLT_HIUPC_MinPixelThrust0p8_MaxPixelCluster10000_v*"]
hltUPCMonopole.throw = False
hltUPCMonopole.andOr = True

from HLTrigger.special.hltPixelActivityFilter_cfi import hltPixelActivityFilter as _hltPixelActivityFilter
hltPixelActivityFilterMinClusters40 = _hltPixelActivityFilter.clone(inputTag = "siPixelClusters", minClusters = 40)

from Configuration.Skimming.PDWG_EXOMONOPOLE_cff import EXOMonopoleSkimContent
upcMonopoleSkimContent = EXOMonopoleSkimContent.clone()
upcMonopoleSkimContent.outputCommands.append('keep FEDRawDataCollection_rawDataRepacker_*_*')

# UPC monopole skim sequence
upcMonopoleSkimSequence = cms.Sequence(hltUPCMonopole * hltPixelActivityFilterMinClusters40)
