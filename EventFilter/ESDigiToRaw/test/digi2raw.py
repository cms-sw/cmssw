import FWCore.ParameterSet.Config as cms

process = cms.Process('ESDIGI2RAW')

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('EventFilter/ESDigiToRaw/esDigiToRaw_cfi')
process.esDigiToRaw.debugMode = True

# Input source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:/home/cmkuo/CMSSW/DQM/CMSSW_3_0_0_pre1/src/PYTHIA6_ZeeJetpt_0_15_10TeV_cff_py_GEN_SIM_DIGI.root')
                            )

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
                                  fileName = cms.untracked.string('digi2raw.root'),
                                  )

process.digi2raw = cms.Path(process.esDigiToRaw)
process.out_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(process.digi2raw, process.out_step)

