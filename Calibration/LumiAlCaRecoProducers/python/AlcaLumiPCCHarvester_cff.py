import FWCore.ParameterSet.Config as cms

from Calibration.LumiAlCaRecoProducers.CorrPCCProducer_cfi import*

DQMStore = cms.Service("DQMStore")

dqmEnvLumiPCC = cms.EDAnalyzer('DQMEventInfo',
                               subSystemFolder=cms.untracked.string('AlCaReco'))


ALCAHARVESTLumiPCC = cms.Sequence(corrPCCProd + dqmEnvLumiPCC)
