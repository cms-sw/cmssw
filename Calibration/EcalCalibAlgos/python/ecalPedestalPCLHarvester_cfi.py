import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

ECALpedestalPCLHarvester = DQMEDHarvester('ECALpedestalPCLHarvester',
                                          MinEntries = cms.int32(100), #skip channel if stat is low
                                          ChannelStatusToExclude = cms.vstring(), # db statuses to exclude
                                          checkAnomalies  = cms.bool(False), # whether or not to avoid creating sqlite file in case of many changed pedestals
                                          nSigma          = cms.double(5.0), #  threshold in sigmas to define a pedestal as anomally changed
                                          thresholdAnomalies = cms.double(0.1),# threshold (fraction of changed pedestals) to avoid creation of sqlite file 
                                          dqmDir      = cms.string('AlCaReco/EcalPedestalsPCL') 
                                          )
