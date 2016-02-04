import FWCore.ParameterSet.Config as cms

process = cms.Process('ESRAW2DIGI')

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('EventFilter/ESRawToDigi/esRawToDigi_cfi')
process.esRawToDigi.sourceTag = 'esDigiToRaw'
process.esRawToDigi.debugMode = True

# Input source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:/home/cmkuo/CMSSW/tmp/CMSSW_3_1_0_pre4/src/EventFilter/ESDigiToRaw/test/digi2raw.root')
                            )

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
                                  fileName = cms.untracked.string('raw2digi.root'),
                                  )

process.raw2digi = cms.Path(process.esRawToDigi)
process.out_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(process.raw2digi, process.out_step)

