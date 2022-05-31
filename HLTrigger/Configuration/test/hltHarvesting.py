import glob
import FWCore.ParameterSet.Config as cms

process = cms.Process('HARVESTING')

# read all the DQMIO files produced by the previous jobs
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring( sorted( "file:%s" % f for f in glob.glob("DQMIO*.root") ) )
)

# load the HLT harvesting sequence
process.load('HLTrigger.Configuration.hltHarvestingSequence_cff')

# DQM file saver
process.load('DQMServices.Components.DQMFileSaver_cfi')
process.dqmSaver.convention   = "Online"
process.dqmSaver.saveByRun    = 1
process.dqmSaver.saveAtJobEnd = False
process.dqmSaver.workflow     = "/DQM/HLT/Online"

process.DQMFileSaverOutput = cms.EndPath( process.HLTHarvestingSequence + process.dqmSaver )
