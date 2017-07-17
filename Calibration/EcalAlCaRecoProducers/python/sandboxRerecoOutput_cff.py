import FWCore.ParameterSet.Config as cms


sandboxRerecoOutputCommands = cms.untracked.vstring( [
    'drop recoGsfElectron*_gsfElectron*_*_*',
    'keep recoGsfElectron*_electronRecalibSCAssociator*_*_*'
        ]
                                                     )
