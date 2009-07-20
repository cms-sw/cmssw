import FWCore.ParameterSet.Config as cms

#------------------------------------------------------------------
# Filter against APV-induced noisy events
#------------------------------------------------------------------

from DPGAnalysis.SiStripTools.eventwithhistoryproducer_cfi import *
from DPGAnalysis.SiStripTools.configurableapvcyclephaseproducer_GR09_cfi import *

from DPGAnalysis.SiStripTools.apvlatency.apvlatencyinfofromconddb_CRAFT_cff import *
essapvlatency.connect = cms.string("sqlite_file:/afs/cern.ch/cms/tracker/sistrlocrec/CRAFTReproIn31X/latency09_fromDPG.db")
#
from DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_HugeEvents_cfi import *
potentialTIBTECHugeEvents.partitionName = cms.untracked.string("TM")
#
from DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_FrameHeaderEvents_firstpeak_cfi import *
potentialTIBTECFrameHeaderEventsFPeak.partitionName = cms.untracked.string("TM")
potentialTIBTECFrameHeaderEventsFPeak.absBXInCycleRangeLtcyAware = cms.untracked.vint32(19,21)
#
potentialTIBTECFrameHeaderEventsAdditionalPeak = potentialTIBTECFrameHeaderEventsFPeak.clone() 
potentialTIBTECFrameHeaderEventsAdditionalPeak.absBXInCycleRangeLtcyAware = cms.untracked.vint32(24,25)
#
from DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_FrameHeaderEvents_widerange_cfi import *
potentialTIBTECFrameHeaderEventsWide.partitionName = cms.untracked.string("TM")
