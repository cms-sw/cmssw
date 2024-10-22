import FWCore.ParameterSet.Config as cms

process = cms.Process('ParticleFlowDQMOffline')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

# my analyzer
process.load('DQMOffline.ParticleFlow.runBasic_cfi')


with open('fileList.log') as f:
    lines = f.readlines()
#Input source
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(lines))


from DQMOffline.ParticleFlow.runBasic_cfi import *

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
                                     fileName = cms.untracked.string("OUT_step1.root"))


process.p = cms.Path(process.PFAnalyzer)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)


## Schedule definition
process.schedule = cms.Schedule(
    process.p,
    process.DQMoutput_step
    )









