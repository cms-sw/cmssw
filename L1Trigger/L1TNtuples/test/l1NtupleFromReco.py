import FWCore.ParameterSet.Config as cms

# make ntuples from RECO (ie. remove RAW)

from L1TriggerDPG.L1Ntuples.l1Ntuple_cfg import *

#process.GlobalTag.globaltag = 'GR10_P_V8::All'
process.GlobalTag.globaltag = 'GR_R_311_V2::All'

#process.p.remove(process.gtDigis)
#process.p.remove(process.gtEvmDigis)
#process.p.remove(process.gctDigis)
#process.p.remove(process.dttfDigis)
#process.p.remove(process.csctfDigis)
#process.p.remove(process.l1extraParticles)

#process.l1NtupleProducer.GMTInputTag = cms.InputTag("none")
#process.l1NtupleProducer.GTEvmInputTag = cms.InputTag("none")
#process.l1NtupleProducer.GTInputTag = cms.InputTag("gtDigis")
#process.l1NtupleProducer.gctCentralJetsSource = cms.InputTag("none","cenJets")
#process.l1NtupleProducer.gctNonIsoEmSource = cms.InputTag("none","nonIsoEm")
#process.l1NtupleProducer.gctForwardJetsSource = cms.InputTag("none","forJets")
#process.l1NtupleProducer.gctIsoEmSource = cms.InputTag("none","isoEm")
#process.l1NtupleProducer.gctEnergySumsSource = cms.InputTag("none","")
#process.l1NtupleProducer.gctTauJetsSource = cms.InputTag("none","tauJets")
#process.l1NtupleProducer.rctSource = cms.InputTag("none")
#process.l1NtupleProducer.dttfSource = cms.InputTag("none")
