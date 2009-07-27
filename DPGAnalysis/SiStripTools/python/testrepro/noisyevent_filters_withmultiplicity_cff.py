import FWCore.ParameterSet.Config as cms

#------------------------------------------------------------------
# Filter against APV-induced noisy events
#------------------------------------------------------------------

from DPGAnalysis.SiStripTools.eventwithhistoryproducer_cfi import *
from DPGAnalysis.SiStripTools.largesistripclusterevents_cfi import *
largeSiStripClusterEvents.collectionName = cms.InputTag("calZeroBiasClusters")
