import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TNtuples.l1NtupleProducer_cfi import *

# get GT and HLT results from HLT digis
l1NtupleProducer.hltSource            = cms.InputTag("TriggerResults::HLT"),
l1NtupleProducer.gtSource             = cms.InputTag("hltGtDigis"),

# turn off everything that is not in the nano DST
l1NtupleProducer.generatorSource      = cms.InputTag("none"),
l1NtupleProducer.simulationSource     = cms.InputTag("none"),
l1NtupleProducer.gmtSource            = cms.InputTag("none"),
l1NtupleProducer.gtEvmSource          = cms.InputTag("none"),
l1NtupleProducer.gctCentralJetsSource = cms.InputTag("none",""),
l1NtupleProducer.gctNonIsoEmSource    = cms.InputTag("none",""),
l1NtupleProducer.gctForwardJetsSource = cms.InputTag("none",""),
l1NtupleProducer.gctIsoEmSource       = cms.InputTag("none",""),
l1NtupleProducer.gctEnergySumsSource  = cms.InputTag("none",""),
l1NtupleProducer.gctTauJetsSource     = cms.InputTag("none",""),
l1NtupleProducer.gctIsoTauJetsSource  = cms.InputTag("none",""),
l1NtupleProducer.rctSource            = cms.InputTag("none"),
l1NtupleProducer.dttfSource           = cms.InputTag("none"),
l1NtupleProducer.ecalSource           = cms.InputTag("none"),
l1NtupleProducer.hcalSource           = cms.InputTag("none"),
l1NtupleProducer.csctfTrkSource       = cms.InputTag("none"),
l1NtupleProducer.csctfLCTSource       = cms.InputTag("none"),
l1NtupleProducer.csctfStatusSource    = cms.InputTag("none"),
l1NtupleProducer.csctfDTStubsSource   = cms.InputTag("none"),
l1NtupleProducer.puMCFile             = cms.untracked.string(""),
l1NtupleProducer.puDataFile           = cms.untracked.string(""),
l1NtupleProducer.puMCHist             = cms.untracked.string(""),
l1NtupleProducer.puDataHist           = cms.untracked.string(""),


