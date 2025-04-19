####################################### BPHnano #####################################
#### Creates custom nanoAOD for multiple B final states. Final states are defined in
#### _cfi.py under python. See bellow for runtime options
## Author: G Karathanasis (gkaratha), CERN


from FWCore.ParameterSet.VarParsing import VarParsing
import FWCore.ParameterSet.Config as cms


def Defaultsamples(isMC,decay):
    if isMC:
       
       if decay=="KLL":
          return ['root://cms-xrd-global.cern.ch//store/mc/Run3Summer22MiniAODv4/BuToJpsiK_BMuonFilter_SoftQCDnonD_TuneCP5_13p6TeV_pythia8-evtgen/MINIAODSIM/130X_mcRun3_2022_realistic_v5-v2/2520000/ed50617a-980e-4b7a-8dc4-892e0f5ebd77.root']
       elif decay=="TrkTrkLL":
           return ['root://cms-xrd-global.cern.ch//store/mc/Run3Summer22EEMiniAODv4/BdToJpsiKstar_BMuonFilter_SoftQCDnonD_TuneCP5_13p6TeV_pythia8-evtgen/MINIAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/2540000/04415d2e-62f7-4c64-aa43-27cd63a43243.root']
       elif decay=="KshortLL":
          return ['root://cms-xrd-global.cern.ch//store/mc/Run3Summer23MiniAODv4/B0ToJpsiK0s_JpsiFilter_MuFilter_K0sFilter_TuneCP5_13p6TeV_pythia8-evtgen/MINIAODSIM/130X_mcRun3_2023_realistic_v14-v3/2820000/02555ce8-49a9-485f-9d46-3c5c49a8359c.root']
       else:
          return ['root://cms-xrd-global.cern.ch//store/mc/Run3Summer22MiniAODv4/BuToJpsiK_BMuonFilter_SoftQCDnonD_TuneCP5_13p6TeV_pythia8-evtgen/MINIAODSIM/130X_mcRun3_2022_realistic_v5-v2/2520000/ed50617a-980e-4b7a-8dc4-892e0f5ebd77.root',\
                  'root://cms-xrd-global.cern.ch//store/mc/Run3Summer22EEMiniAODv4/BdToJpsiKstar_BMuonFilter_SoftQCDnonD_TuneCP5_13p6TeV_pythia8-evtgen/MINIAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/2540000/04415d2e-62f7-4c64-aa43-27cd63a43243.root',\
                  'root://cms-xrd-global.cern.ch//store/mc/Run3Summer23MiniAODv4/B0ToJpsiK0s_JpsiFilter_MuFilter_K0sFilter_TuneCP5_13p6TeV_pythia8-evtgen/MINIAODSIM/130X_mcRun3_2023_realistic_v14-v3/2820000/02555ce8-49a9-485f-9d46-3c5c49a8359c.root']
    else:
       return ['root://cms-xrd-global.cern.ch//store/data/Run2024C/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/379/415/00000/b40397b5-61c6-4887-8f4e-025e8ca925ee.root']


options = VarParsing('python')

options.register('globalTag', '130X_dataRun3_Prompt_v3', 
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "Global tag"
)

options.register('isMC', False,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.bool,
    "Adds gen info/matching"
)


options.register('wantSummary', True,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.bool,
    "Processing summary"
)

options.register('wantFullRECO', False,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.bool,
    "Produces additional EDM file"
    )

options.register('reportEvery', 100,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.int,
    "Report every N events"
)

options.register('skip', 0,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.int,
    "Skip first N events"
)

options.register('decay', 'all',
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "Options: all KLL KshortLL TrkTrkLL"
)


options.setDefault('maxEvents', -1)
options.setDefault('tag', 'test')

#print(options)
options.parseArguments()
#print("////////////////// BPHnano running with options: ////////////////////////")
#print(options)
#print("/////////////////////////////////////////////////////////////////////////")

globaltag = '124X_mcRun3_2022_realistic_v11' if options.isMC else '130X_dataRun3_Prompt_v3'


if options.isMC:
   options.tag+="_mc"
else:
   options.tag+="_data"

options.tag+='_'
options.tag+=options.decay

outputFileNANO = cms.untracked.string('_'.join(['bph_nano',options.tag])+'.root')
outputFileFEVT = cms.untracked.string('_'.join(['bph_edm',options.tag])+'.root')


if not options.inputFiles:
   options.inputFiles = Defaultsamples(options.isMC,options.decay)

annotation = '%s nevts:%d' % (outputFileNANO, options.maxEvents)



# Process
from Configuration.StandardSequences.Eras import eras
#from PhysicsTools.NanoAOD.nano_eras_cff import Run3_2023

process = cms.Process('BPHNANO',eras.Run3,eras.run3_nanoAOD_pre142X)

# import of standard configurations
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('PhysicsTools.BPHNano.nanoBPH_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.MessageLogger.cerr.FwkReport.reportEvery = options.reportEvery
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

# Input source
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles),
    secondaryFileNames = cms.untracked.vstring(),
    skipEvents=cms.untracked.uint32(options.skip),
)

process.options = cms.untracked.PSet(
#    TryToContinue = cms.untracked.vstring('ProductNotFound'),
    wantSummary = cms.untracked.bool(options.wantSummary),
)

process.nanoMetadata.strings.tag = annotation
# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string(annotation),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition
process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = outputFileFEVT,
    outputCommands = (cms.untracked.vstring('keep *',
                                            'drop *_*_SelectedTransient*_*',
                     )),
    splitLevel = cms.untracked.int32(0)
)

process.NANOAODoutput = cms.OutputModule("NanoAODOutputModule",
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(9),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('NANOAOD'),
        filterName = cms.untracked.string('')
    ),
    fileName = outputFileNANO,
    outputCommands = cms.untracked.vstring(
      'drop *',
      "keep nanoaodFlatTable_*Table_*_*",     # event data
      "keep nanoaodUniqueString_nanoMetadata_*_*",   # basic metadata
#      "keep edmTriggerResults_*_*_*",
    )

)


# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, globaltag, '')
from PhysicsTools.BPHNano.nanoBPH_cff import *
if options.isMC:
   process = nanoAOD_customizeMC(process)

#process = nanoAOD_customizeCommon(process)
process = nanoAOD_customizeMuonBPH(process)#,options.isMC)
process = nanoAOD_customizeDiMuonBPH(process)#,options.isMC)
process = nanoAOD_customizeTrackBPH(process)#,options.isMC)

if options.decay == "KLL":
   process = nanoAOD_customizeBToKLL(process)#,options.isMC)

elif options.decay == "TrkTrkLL":
   process = nanoAOD_customizeBToTrkTrkLL(process)#,options.isMC)

elif options.decay == "KshortLL": 
   process = nanoAOD_customizeBToKshortLL(process)#,options.isMC)

elif options.decay =="Kshort_K_LL":
   process = nanoAOD_customizeBToKLL(process)#,options.isMC)
   process = nanoAOD_customizeBToKshortLL(process)#,options.isMC)
    
elif options.decay == "all":
   process = nanoAOD_customizeBPH(process)#,options.isMC)

else:
#   print("Undefined decay option")
   import sys
   sys.exit(1)

#print("processing modules:",process.nanoSequence)

process.nanoAOD_BPH_step = cms.Path(process.nanoSequence)


process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
process.NANOAODoutput_step = cms.EndPath(process.NANOAODoutput)


process.schedule = cms.Schedule(
    process.nanoAOD_BPH_step,
    process.endjob_step,
    process.NANOAODoutput_step
    )

if options.wantFullRECO:
   process.schedule.insert(0,process.FEVTDEBUGHLToutput_step)

from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

process.NANOAODoutput.SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring(
        'nanoAOD_BPH_step',
    )
)

### from https://hypernews.cern.ch/HyperNews/CMS/get/physics-validation/3287/1/1/1/1/1.html
process.add_(cms.Service('InitRootHandlers', EnableIMT = cms.untracked.bool(False)))
process.NANOAODoutput.fakeNameForCrab=cms.untracked.bool(True)

process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)

#print(process.dumpPython())


