import FWCore.ParameterSet.Config as cms

#------------------------------------------------------------------
# Filter against APV-induced noisy events
#------------------------------------------------------------------

from myTools.HistoricizedEventProducer.historicizedeventproducer_cfi import *
from myTools.SimpleAPVPhaseProducer.simpleapvphaseproducer_CRAFT_cfi import *
#from myTools.SimpleAPVPhaseProducer.configurableapvphaseproducer_CRAFT08_cfi import *

from myTools.APVLatencyInfoESSource.apvlatencyinfofromconddb_CRAFT_cff import *
essapvlatency.connect = cms.string("sqlite_file:/afs/cern.ch/cms/tracker/sistrlocrec/CRAFTReproIn31X/latency_31X.db")

from myTools.HistoricizedEventFilter.Potential_TOB_HugeEvents_cfi import *
from myTools.HistoricizedEventFilter.Potential_TIBTEC_HugeEvents_cfi import *

from myTools.HistoricizedEventFilter.Potential_TOB_FrameHeaderEvents_firstpeak_cfi import *
potentialTOBFrameHeaderEventsFPeak.absBXInCycleRangeLtcyAware = cms.untracked.vint32(19,21)

from myTools.HistoricizedEventFilter.Potential_TIBTEC_FrameHeaderEvents_firstpeak_cfi import *
potentialTIBTECFrameHeaderEventsFPeak.absBXInCycleRangeLtcyAware = cms.untracked.vint32(19,21)

potentialTOBFrameHeaderEventsAdditionalPeak = potentialTOBFrameHeaderEventsFPeak.clone() 
potentialTOBFrameHeaderEventsAdditionalPeak.absBXInCycleRangeLtcyAware = cms.untracked.vint32(24,25)

potentialTIBTECFrameHeaderEventsAdditionalPeak = potentialTIBTECFrameHeaderEventsFPeak.clone() 
potentialTIBTECFrameHeaderEventsAdditionalPeak.absBXInCycleRangeLtcyAware = cms.untracked.vint32(24,25)
