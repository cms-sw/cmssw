import FWCore.ParameterSet.Config as cms

# Tracker
from DQM.SiPixelHeterogeneous.siPixelTrackComparisonHarvester_cfi import *
hltSiPixelTrackComparisonHarvester = siPixelTrackComparisonHarvester.clone(topFolderName = 'HLT/HeterogeneousComparisons/PixelTracks')

HLTHeterogeneousMonitoringHarvesting =  cms.Sequence(hltSiPixelTrackComparisonHarvester)
