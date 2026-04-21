import FWCore.ParameterSet.Config as cms

from HLTrigger.special.hltPixelThrustFilter_cfi import hltPixelThrustFilter as _hltPixelThrustFilter
hltUPCMonopole = _hltPixelThrustFilter.clone(inputTag = "siPixelClusters", maxNPixels = 10000, minNSaturatedPixels = 2, minThrust = 0.85)

from Configuration.Skimming.PDWG_EXOMONOPOLE_cff import EXOMonopoleSkimContent
upcMonopoleSkimContent = EXOMonopoleSkimContent.clone()
upcMonopoleSkimContent.outputCommands.append('keep FEDRawDataCollection_rawDataRepacker_*_*')

# UPC monopole skim sequence
upcMonopoleSkimSequence = cms.Sequence(hltUPCMonopole)
