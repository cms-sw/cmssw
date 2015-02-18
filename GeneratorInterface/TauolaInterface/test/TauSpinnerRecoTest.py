import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions#Global_Tags_for_Monte_Carlo_Prod
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("GeneratorInterface.TauolaInterface.TauSpinner_cfi")

process.MessageLogger.cerr = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(-1))
    )

numberOfEvents = 10

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
                                                   TauSpinnerReco = cms.PSet(
    initialSeed = cms.untracked.uint32(123456789),
    engineName = cms.untracked.string('HepJamesRandom')
    )
                                                   )
process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

process.GlobalTag.globaltag = 'MC_50_V13::All'

process.source = cms.Source("PoolSource",fileNames = cms.untracked.vstring('file:003320CA-478C-E411-8482-0025905A611E.root'))#0024383A-A989-E411-A5D2-0025905A60E4.root'))

process.debugOutput = cms.OutputModule("PoolOutputModule",
                                       outputCommands = cms.untracked.vstring('keep *'),
                                       fileName = cms.untracked.string('TauSpinerRecoTest.root'),
                                       )
process.out_step = cms.EndPath(process.debugOutput)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(numberOfEvents) )
process.p1 = cms.Path(process.TauSpinnerReco )
process.schedule = cms.Schedule(process.p1)
process.schedule.append(process.out_step)
