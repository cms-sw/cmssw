import FWCore.ParameterSet.Config as cms

from DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_FrameHeaderEvents_additionalpeak_AlCaReco_cfi import PotentialTIBTECFrameHeaderEventsAdditionalPeak as alcareco1
alcareco1.commonConfiguration.historyProduct = cms.untracked.InputTag("consecutiveHEs")
alcareco1.commonConfiguration.APVPhaseLabel = cms.untracked.string("APVPhases")

from DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_FrameHeaderEvents_firstpeak_AlCaReco_cfi import PotentialTIBTECFrameHeaderEventsFPeak as alcareco2
alcareco2.commonConfiguration.historyProduct = cms.untracked.InputTag("consecutiveHEs")
alcareco2.commonConfiguration.APVPhaseLabel = cms.untracked.string("APVPhases")

from DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_HugeEvents_AlCaReco_cfi import PotentialTIBTECHugeEvents as alcareco3
alcareco3.commonConfiguration.historyProduct = cms.untracked.InputTag("consecutiveHEs")
alcareco3.commonConfiguration.APVPhaseLabel = cms.untracked.string("APVPhases")

import DPGAnalysis.SiStripTools.eventtimedistribution_cfi

etdalca1 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()
etdalca2 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()
etdalca3 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()
etdalca4 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()

alcas1 = cms.Sequence(alcareco1 + etdalca1)
alcas2 = cms.Sequence(alcareco2 + etdalca2)
alcas3 = cms.Sequence(alcareco3 + etdalca3)
alcas4 = cms.Sequence(~alcareco1 + ~alcareco2 + ~alcareco3 + etdalca4)




