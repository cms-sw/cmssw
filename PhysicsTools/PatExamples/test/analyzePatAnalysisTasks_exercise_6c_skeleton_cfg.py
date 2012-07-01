## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

## dp^?o PF2PAT
from PhysicsTools.PatAlgos.tools.pfTools import *

usePF2PAT(process,runPF2PAT=True, jetAlgo='AK5', runOnMC=True)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.MessageLogger = cms.Service("MessageLogger")

process.TFileService = cms.Service("TFileService",
  fileName = cms.string('TFileServiceOutput.root')
)
process.maxEvents.input = 100
process.p = cms.Path(
#    process.patDefaultSequence  +
    getattr(process,"patPF2PATSequence")
#    second PF2PAT
#    + getattr(process,"patPF2PATSequence"+postfix2)
)
#########################
## This tool needs some more things to be there:
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = cms.string( autoCond[ 'startup' ] )
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("PhysicsTools.PatAlgos.patSequences_cff")

process.load("PhysicsTools.PatUtils.patPFMETCorrections_cff")


process.p.insert(0,process.type0PFMEtCorrection  *  process.patPFMETtype0Corr)

process.options.wantSummary = False

