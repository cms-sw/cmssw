import FWCore.ParameterSet.Config as cms

process.generator = cms.EDFilter("Herwig7GeneratorFilter",
    run = cms.string('InterfaceMatchboxTest'),
    repository = cms.string('${HERWIGPATH}/HerwigDefaults.rpo'),
    dataLocation = cms.string('${HERWIGPATH:-6}'),
    generatorModule = cms.string('/Herwig/Generators/EventGenerator'),
    eventHandlers = cms.string('/Herwig/EventHandlers'),
    configFiles = cms.vstring(),
    crossSection = cms.untracked.double(-1),
    parameterSets = cms.vstring(),
    filterEfficiency = cms.untracked.double(1.0),
    Matchbox = cms.vstring( 'read LHE.in',
        'set LesHouchesReader:FileName ${CMSSW_BASE}/src/cmsgrid_final.lhe.gz'
        )
)

process.ProductionFilterSequence = cms.Sequence(process.generator)