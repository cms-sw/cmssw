import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
import FWCore.ParameterSet.VarParsing as VarParsing
from Configuration.AlCa.GlobalTag import GlobalTag

import sys

process = cms.Process('MANUALRECO', eras.Run3)

#SETUP PARAMETERS
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    FailPath = cms.untracked.vstring('Type Mismatch') # not crashing on this exception type
    )
options = VarParsing.VarParsing ('analysis')
options.register('outputFileName',
                'PPS_ALCARECO.root',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "output ROOT file name")
options.register('useJsonFile',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.bool,
                "Do not use JSON file")
options.register('jsonFileName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "JSON file list name")
options.register('globalTag',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "GT to use")
options.parseArguments()

#SETUP LOGGER
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet( 
        optionalPSet = cms.untracked.bool(True),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(False),
        FwkReport = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            reportEvery = cms.untracked.int32(100),
            limit = cms.untracked.int32(50000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('INFO')
    ),
    categories = cms.untracked.vstring(
        "FwkReport"
    ),
)


#CONFIGURE PROCESS
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20000) )

#SETUP GLOBAL TAG
if options.globalTag != '':
    gt = options.globalTag
else:
    gt = 'auto:run3_data_prompt'


process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
print('Using GT:',gt)
process.GlobalTag = GlobalTag(process.GlobalTag, gt)

# Manually add local alignment from sqlite file
# process.GlobalTag.toGet = cms.VPSet(
#     cms.PSet(
#         record = cms.string('RPRealAlignmentRecord'),
#         tag = cms.string("CTPPSRPAlignment_real"),
#         label = cms.untracked.string(""),
#         connect = cms.string("sqlite_file:/eos/cms/store/group/phys_pps/reconstruction/alignment/2023/CTPPSRPAlignment_reference.db")
#         )
# )

if len(options.inputFiles) != 0:
    inputFiles = cms.untracked.vstring(options.inputFiles)
else:
    # Example input file
    inputFiles = cms.untracked.vstring(
        [
            '/store/data/Run2023C/AlCaPPSPrompt/ALCARECO/PPSCalMaxTracks-PromptReco-v3/000/367/696/00000/10d6bcd2-a5eb-4fd3-bc30-daeae548887e.root',
        ]
    )
    
#SETUP INPUT
print('Input files:\n',inputFiles, sep='')
process.source = cms.Source("PoolSource",
    fileNames = inputFiles,
    # Drop everything from the prompt alcareco besides the digis at input
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_*_*_RECO',
        'keep *_*Digi*_*_RECO'
    )
)

if options.useJsonFile == True:
    import FWCore.PythonUtilities.LumiList as LumiList
    jsonFileName = options.jsonFileName
    print("Using JSON file:",jsonFileName)
    process.source.lumisToProcess = LumiList.LumiList(filename = jsonFileName).getVLuminosityBlockRange()

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(options.outputFileName),
    # Keep only the new products
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_*_*_RECO"
    )
)

# Load the ALCARECO reco step from DIGI
process.load("Calibration.PPSAlCaRecoProducer.ALCARECOPPSCalMaxTracks_cff")
# Remove sampic reco
process.recoPPSSequenceAlCaRecoProducer.remove(process.diamondSampicLocalReconstructionTaskAlCaRecoProducer)

# processing sequences
process.path = cms.Path(
    process.recoPPSSequenceAlCaRecoProducer
)

process.end_path = cms.EndPath(
  process.output
)