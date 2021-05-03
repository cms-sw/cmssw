import FWCore.ParameterSet.Config as cms


generator = cms.EDFilter("Herwig7GeneratorFilter",
    run = cms.string('InterfaceMatchboxTest'),
    repository = cms.string('${HERWIGPATH}/HerwigDefaults.rpo'),
    dataLocation = cms.string('${HERWIGPATH:-6}'),
    generatorModule = cms.string('/Herwig/Generators/EventGenerator'),
    eventHandlers = cms.string('/Herwig/EventHandlers'),
    configFiles = cms.vstring('LHC-Matchbox.in'),
    crossSection = cms.untracked.double(-1),
    parameterSets = cms.vstring(),
    filterEfficiency = cms.untracked.double(1.0),
)

ProductionFilterSequence = cms.Sequence(generator)
