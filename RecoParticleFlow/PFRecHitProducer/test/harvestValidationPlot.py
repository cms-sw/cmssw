# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step6 --conditions auto:phase1_2022_realistic -s HARVESTING:@pfDQM --era Run3 --filetype DQM --filein file:step5.root --fileout file:step6.root
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('HARVESTING',Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')

process.maxEvents.input = 1

# Input source
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring('file:DQMIO.root')
)

process.dqmsave_step = cms.Path(process.DQMSaver)
process.schedule = cms.Schedule(process.dqmsave_step)
