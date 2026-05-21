import FWCore.ParameterSet.Config as cms

# module
zdcEnergyFilter1nOr = cms.EDFilter(
    'HiZDCFilter',
    ZDCRecHitSource = cms.InputTag('zdcrecoRun3'),
    threshold4ltPlus = cms.double(1500), # threshold for less than for ZDC+
    threshold4gtPlus = cms.double(900), # threshold for greater than for ZDC+
    threshold4ltMinus = cms.double(1500), # threshold for less than for ZDC-
    threshold4gtMinus = cms.double(900), # threshold for greater than for ZDC-
    algorithm = cms.string('gt OR') # [lt/gt][AND/OR], XOR (not case-sensitive)
)
zdcEnergyFilter0nOr = zdcEnergyFilter1nOr.clone( algorithm = 'lt OR' )
zdcEnergyFilter0nAnd = zdcEnergyFilter1nOr.clone( algorithm = 'lt AND' )
zdcEnergyFilterXOr = zdcEnergyFilter1nOr.clone( algorithm = 'XOR' )

# path
pzdcEnergyFilter1nOr = cms.Path(zdcEnergyFilter1nOr)
pzdcEnergyFilter0nOr = cms.Path(zdcEnergyFilter0nOr)
pzdcEnergyFilter0nAnd = cms.Path(zdcEnergyFilter0nAnd)
pzdcEnergyFilterXOr = cms.Path(zdcEnergyFilterXOr)
