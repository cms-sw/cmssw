import FWCore.ParameterSet.Config as cms

#---------------------------------------------
# Filters to remove APV induced noisy events #
#---------------------------------------------

from DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_HugeEvents_AlCaReco_cfi import *

from DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_FrameHeaderEvents_firstpeak_AlCaReco_cfi import *

from DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_FrameHeaderEvents_additionalpeak_AlCaReco_cfi import *

#--------------------------------------------
# Filter to remove high multiplicity events #
#--------------------------------------------

from DPGAnalysis.SiStripTools.largesistripclusterevents_AlCaReco_cfi import *

#------------
# Sequences #
#------------

seqAPVCycleFilter = cms.Sequence(~PotentialTIBTECHugeEvents*
                                 ~PotentialTIBTECFrameHeaderEventsFPeak*
                                 ~PotentialTIBTECFrameHeaderEventsAdditionalPeak)

seqMultiplicityFilter = cms.Sequence(~LargeSiStripClusterEvents)

seqFilters = cms.Sequence(seqAPVCycleFilter*
                          seqMultiplicityFilter)
