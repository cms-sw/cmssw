import FWCore.ParameterSet.Config as cms

#LHE content
GeneratorInterfaceLHE = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep LHERunInfoProduct_*_*_*',
        'keep LHEEventProduct_*_*_*',
        'keep *_externalLHEProducer_LHEScriptOutput_*'
    )
)

#RAW content 
GeneratorInterfaceRAW = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep LHERunInfoProduct_*_*_*',
        'keep LHEEventProduct_*_*_*',
        'keep GenRunInfoProduct_generator_*_*',
        'keep GenLumiInfoHeader_generator_*_*',
        'keep GenLumiInfoProduct_generator_*_*',
        'keep GenEventInfoProduct_generator_*_*',
        'keep edmHepMCProduct_generatorSmeared_*_*',
        'keep GenFilterInfo_*_*_*',
        'keep *_genParticles_*_*'
    )
)

#RECO content
GeneratorInterfaceRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep LHERunInfoProduct_*_*_*',
        'keep LHEEventProduct_*_*_*',
        'keep GenRunInfoProduct_generator_*_*',
        'keep GenLumiInfoHeader_generator_*_*',
        'keep GenLumiInfoProduct_generator_*_*',
        'keep GenEventInfoProduct_generator_*_*',
        'keep edmHepMCProduct_generatorSmeared_*_*',
        'keep GenFilterInfo_*_*_*',
        'keep *_genParticles_*_*'
    )
)

#AOD content
GeneratorInterfaceAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep LHERunInfoProduct_*_*_*',
        'keep LHEEventProduct_*_*_*',
        'keep GenRunInfoProduct_generator_*_*',
        'keep GenLumiInfoHeader_generator_*_*',
        'keep GenLumiInfoProduct_generator_*_*',
        'keep GenEventInfoProduct_generator_*_*',
        'keep GenFilterInfo_*_*_*',
        'keep *_genParticles_*_*'
    )
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(GeneratorInterfaceRAW,
                     outputCommands = GeneratorInterfaceRAW.outputCommands+['keep edmHepMCProduct_LHCTransport_*_*'])
run3_common.toModify(GeneratorInterfaceRECO,
                     outputCommands = GeneratorInterfaceRECO.outputCommands+['keep edmHepMCProduct_LHCTransport_*_*'])
