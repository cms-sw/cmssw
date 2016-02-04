import FWCore.ParameterSet.Config as cms

process = cms.Process('ESDIGI2RAW')

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('EventFilter/ESDigiToRaw/esDigiToRaw_cfi')
process.esDigiToRaw.debugMode = True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
    )

# Input source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:/home/cmkuo/CMSSW/tmp/CMSSW_3_1_0_pre2/src/2C187380-6203-DE11-9794-001617C3B76E.root')
                            )

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
                                  fileName = cms.untracked.string('digi2raw.root'),
                                  )

process.digi2raw = cms.Path(process.esDigiToRaw)
process.out_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(process.digi2raw, process.out_step)

