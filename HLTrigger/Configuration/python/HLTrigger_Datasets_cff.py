# /dev/CMSSW_3_6_0/pre4/HIon/V18

import FWCore.ParameterSet.Config as cms

# dump of the Stream A Datasets defined in the HLT table

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as JetMETTauMonitor
JetMETTauMonitor.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
JetMETTauMonitor.l1tResults = cms.InputTag('')
JetMETTauMonitor.throw      = cms.bool(False)
JetMETTauMonitor.triggerConditions = cms.vstring()

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as EGMonitor
EGMonitor.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
EGMonitor.l1tResults = cms.InputTag('')
EGMonitor.throw      = cms.bool(False)
EGMonitor.triggerConditions = cms.vstring()

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as EG
EG.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
EG.l1tResults = cms.InputTag('')
EG.throw      = cms.bool(False)
EG.triggerConditions = cms.vstring()

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as RandomTriggers
RandomTriggers.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
RandomTriggers.l1tResults = cms.InputTag('')
RandomTriggers.throw      = cms.bool(False)
RandomTriggers.triggerConditions = cms.vstring()

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as HcalHPDNoise
HcalHPDNoise.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
HcalHPDNoise.l1tResults = cms.InputTag('')
HcalHPDNoise.throw      = cms.bool(False)
HcalHPDNoise.triggerConditions = cms.vstring()

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as ZeroBias
ZeroBias.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
ZeroBias.l1tResults = cms.InputTag('')
ZeroBias.throw      = cms.bool(False)
ZeroBias.triggerConditions = cms.vstring()

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as MuMonitor
MuMonitor.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
MuMonitor.l1tResults = cms.InputTag('')
MuMonitor.throw      = cms.bool(False)
MuMonitor.triggerConditions = cms.vstring()

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as Cosmics
Cosmics.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
Cosmics.l1tResults = cms.InputTag('')
Cosmics.throw      = cms.bool(False)
Cosmics.triggerConditions = cms.vstring()

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as Mu
Mu.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
Mu.l1tResults = cms.InputTag('')
Mu.throw      = cms.bool(False)
Mu.triggerConditions = cms.vstring()

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as JetMETTau
JetMETTau.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
JetMETTau.l1tResults = cms.InputTag('')
JetMETTau.throw      = cms.bool(False)
JetMETTau.triggerConditions = cms.vstring()

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as MinimumBias
MinimumBias.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
MinimumBias.l1tResults = cms.InputTag('')
MinimumBias.throw      = cms.bool(False)
MinimumBias.triggerConditions = cms.vstring()

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as HcalNZS
HcalNZS.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
HcalNZS.l1tResults = cms.InputTag('')
HcalNZS.throw      = cms.bool(False)
HcalNZS.triggerConditions = cms.vstring()

