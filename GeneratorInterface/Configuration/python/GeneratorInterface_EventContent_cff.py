import FWCore.ParameterSet.Config as cms

#RAW content 
GeneratorInterfaceRAW = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep LHERunInfoProduct_source_*_*',
        'keep LHEEventProduct_source_*_*',
        'keep GenRunInfoProduct_generator_*_*',
        'keep GenEventInfoProduct_generator_*_*',
        'keep edmHepMCProduct_generator_*_*',
        'keep GenFilterInfo_*_*_*',
        'keep *_genParticles_*_*'
    )
)

#RECO content
GeneratorInterfaceRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep LHERunInfoProduct_source_*_*',
        'keep LHEEventProduct_source_*_*',
        'keep GenRunInfoProduct_generator_*_*',
        'keep GenEventInfoProduct_generator_*_*',
        'keep edmHepMCProduct_generator_*_*',
        'keep GenFilterInfo_*_*_*',
        'keep *_genParticles_*_*'
    )
)

#AOD content
GeneratorInterfaceAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep LHERunInfoProduct_source_*_*',
        'keep LHEEventProduct_source_*_*',
        'keep GenRunInfoProduct_generator_*_*',
        'keep GenEventInfoProduct_generator_*_*',
        'keep GenFilterInfo_*_*_*',
        'keep *_genParticles_*_*'
    )
)
