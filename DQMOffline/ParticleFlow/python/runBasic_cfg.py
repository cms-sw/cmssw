import FWCore.ParameterSet.Config as cms

# jet calibration stuff
from JetMETCorrections.Configuration.JetCorrectors_cff import ak4PFPuppiL1FastL2L3ResidualCorrectorChain,ak4PFPuppiL1FastL2L3ResidualCorrector,ak4PFPuppiL1FastL2L3Corrector,ak4PFPuppiResidualCorrector,ak4PFPuppiL3AbsoluteCorrector,ak4PFPuppiL2RelativeCorrector,ak4PFPuppiL1FastjetCorrector

dqmAk4PFPuppiL1FastL2L3ResidualCorrector = ak4PFPuppiL1FastL2L3ResidualCorrector.clone()

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


from DQMOffline.ParticleFlow.runBasic_cfi import *

# back to original script
with open('fileList_2.log') as f:
    lines = f.readlines()
#Input source
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(lines))


from DQMOffline.ParticleFlow.runBasic_cfi import *

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
                                     fileName = cms.untracked.string("OUT_step1.root"))


process.p = cms.Path(process.PFAnalyzer)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

dqmAk4PFPuppiL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    dqmAk4PFPuppiL1FastL2L3ResidualCorrector*PFAnalyzer
)

## Schedule definition
process.schedule = cms.Schedule(
    process.p,
    process.DQMoutput_step
    )









