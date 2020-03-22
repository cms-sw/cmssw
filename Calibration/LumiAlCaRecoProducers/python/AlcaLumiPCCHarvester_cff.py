import FWCore.ParameterSet.Config as cms

from Calibration.LumiAlCaRecoProducers.CorrPCCProducer_cfi import*

DQMStore = cms.Service("DQMStore")

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
dqmEnvLumiPCC = DQMEDHarvester('DQMHarvestingMetadata',
                              subSystemFolder=cms.untracked.string('AlCaReco'))


ALCAHARVESTLumiPCC = cms.Sequence(corrPCCProd + dqmEnvLumiPCC)
