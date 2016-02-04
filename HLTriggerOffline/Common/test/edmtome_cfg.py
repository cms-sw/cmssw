# extracts histograms from edm file (hltvalidation_DQM.root)
# to produce the validation edm file
#   cmsDriver.py hltvalidation -s DQM --filein file:input.root
#                                     --number -1 --no-exec
# this generates _cfg file from
#   Configuration/StandardSequences/python/Validation_cff.py
# which includes the hlt validation sequence
#   HLTriggerOffline/Common/python/HLTValidation_cff.py

import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")
process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.MessageLogger = cms.Service("MessageLogger")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:hltvalidation_DQM.root')
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_EDMtoMEConverter_*_*'),
    fileName = cms.untracked.string('out.root')
)

process.p = cms.Path(process.EDMtoMEConverter*process.dqmSaver)
process.outpath = cms.EndPath(process.out)
process.DQMStore.referenceFileName = ''
process.dqmSaver.convention = 'RelVal'
process.dqmSaver.workflow = '/ConverterTester/Test/RECO'
