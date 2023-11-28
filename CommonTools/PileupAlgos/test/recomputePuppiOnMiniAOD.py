# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step5 --conditions auto:run2_mc --nThreads 2 -n 10 --era Run2_2016 --eventcontent MINIAODSIM --filein file:step3_inMINIAODSIM.root -s NANO --datatier NANOAODSIM --mc --fileout file:step5.root
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016

process = cms.Process('NANO',Run2_2016)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/h/hinzmann/stable_13TeV/jetmet/mtd/CMSSW_12_0_0_pre3/src/25202.0_TTbar_13+TTbar_13+DIGIUP15_PU25+RECOUP15_PU25+HARVESTUP15_PU25+NANOUP15_PU25/step3_inMINIAODSIM.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    TryToContinue = cms.untracked.vstring(),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(

        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(1)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step5 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.MINIAODSIMoutput = cms.OutputModule("PoolOutputModule",
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(9),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('MINIAODSIM'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:updatedMINIAODSIM.root'),
    outputCommands = process.MINIAODSIMEventContent.outputCommands
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Rerun PUPPI MET and ak4 jets (but not ak8)
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask
task = getPatAlgosToolsTask(process)
from PhysicsTools.PatAlgos.slimming.puppiForMET_cff import makePuppiesFromMiniAOD
makePuppiesFromMiniAOD(process,True)
process.puppi.useExistingWeights = False
process.puppiNoLep.useExistingWeights = False
from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD
runMetCorAndUncFromMiniAOD(process,isData=False,metType="Puppi",postfix="Puppi",jetFlavor="AK4PFPuppi",recoMetFromPFCs=True)
from PhysicsTools.PatAlgos.patPuppiJetSpecificProducer_cfi import patPuppiJetSpecificProducer
addToProcessAndTask('patPuppiJetSpecificProducer', patPuppiJetSpecificProducer.clone(src=cms.InputTag("patJetsPuppi")), process, task)
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
updateJetCollection(
   process,
   labelName = 'PuppiJetSpecific',
   jetSource = cms.InputTag('patJetsPuppi'),
)
process.updatedPatJetsPuppiJetSpecific.userData.userFloats.src = ['patPuppiJetSpecificProducer:puppiMultiplicity', 'patPuppiJetSpecificProducer:neutralPuppiMultiplicity', 'patPuppiJetSpecificProducer:neutralHadronPuppiMultiplicity', 'patPuppiJetSpecificProducer:photonPuppiMultiplicity', 'patPuppiJetSpecificProducer:HFHadronPuppiMultiplicity', 'patPuppiJetSpecificProducer:HFEMPuppiMultiplicity' ]
addToProcessAndTask('slimmedJetsPuppi', process.updatedPatJetsPuppiJetSpecific.clone(), process, task)
del process.updatedPatJetsPuppiJetSpecific
process.puppiSequence = cms.Sequence(process.puppiMETSequence+process.fullPatMetSequencePuppi+process.patPuppiJetSpecificProducer+process.slimmedJetsPuppi)

# Example how to not use vertex fit for track-vertex-association, but only dz
#process.packedPrimaryVertexAssociationJME.assignment.useVertexFit = False
#process.packedPrimaryVertexAssociationJME.assignment.maxDzSigForPrimaryAssignment = 5.0
#process.packedPrimaryVertexAssociationJME.assignment.maxDzForPrimaryAssignment = 0.1
#process.packedPrimaryVertexAssociationJME.assignment.maxDzErrorForPrimaryAssignment = 0.05
#process.packedPrimaryVertexAssociationJME.assignment.OnlyUseFirstDz = False
#process.pfCHS.vertexAssociationQuality = 6

# Path and EndPath definitions
process.puppi_step = cms.Path(process.puppiSequence)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.MINIAODSIMoutput_step = cms.EndPath(process.MINIAODSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.puppi_step,process.endjob_step,process.MINIAODSIMoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(2)
process.options.numberOfStreams=cms.untracked.uint32(0)
process.options.numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(1)

# customisation of the process.

# End of customisation functions

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
