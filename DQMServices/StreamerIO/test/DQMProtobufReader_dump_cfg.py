# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: test_11_a_1 -s RAW2DIGI,RECO,DQM -n 100 --eventcontent DQM --conditions auto:com10 --filein file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root --data --customise DQMServices/Components/test/customDQM.py --no_exec --python_filename=test_11_a_1.py
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process('RECO')

# import of standard configurations
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
)

from DQMServices.StreamerIO.DQMProtobufReader_cff import DQMProtobufReader
process.source = DQMProtobufReader

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring("cerr"),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
    ),
    #debugModules = cms.untracked.vstring('*'),
)

# Input source
#print dir(process)
#process.source = process.DQMStreamerReader

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('test_11_a_1 nevts:100'),
    name = cms.untracked.string('Applications')
)

process.dqmsave_step = cms.Path(process.DQMSaver)

process.analyzer= cms.EDAnalyzer("DQMStoreAnalyzer");
process.p = cms.Path(process.analyzer)

process.options = cms.untracked.PSet( SkipEvent = cms.untracked.vstring('ProductNotFound') )
process.endjob_step = cms.EndPath(process.endOfProcess)
process.schedule = cms.Schedule(process.p,process.endjob_step,process.dqmsave_step,)
