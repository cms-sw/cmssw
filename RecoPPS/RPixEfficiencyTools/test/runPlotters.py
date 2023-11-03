import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
import FWCore.ParameterSet.VarParsing as VarParsing

import sys

process = cms.Process('PLOTTER', eras.Run3)

#SETUP PARAMETERS
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    FailPath = cms.untracked.vstring('Type Mismatch') # not crashing on this exception type
    )
options = VarParsing.VarParsing ('analysis')
options.register('outputFileName',
                'PPS_ALCARECO_plots',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "output ROOT file name")
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

#CONFIGURE PROCESS
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

#SETUP GLOBAL TAG
from Configuration.AlCa.GlobalTag import GlobalTag
if options.globalTag != '':
    gt = options.globalTag
else:
    gt = 'auto:run3_data_prompt'

# Load geometry from DB
process.load('Geometry.VeryForwardGeometry.geometryRPFromDB_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
print('Using GT:',gt)
process.GlobalTag = GlobalTag(process.GlobalTag, gt)

alcarecoSuffix = ''
# Uncomment the line below to run on ALCARECO files
alcarecoSuffix += 'AlCaRecoProducer'

process.load
process.load('Validation.CTPPS.ctppsLHCInfoPlotter_cfi')
process.ctppsLHCInfoPlotter.outputFile = options.outputFileName + '_lhcInfo.root'

process.ctppsTrackDistributionPlotter = cms.EDAnalyzer("CTPPSTrackDistributionPlotter",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"+alcarecoSuffix),
    outputFile = cms.string(options.outputFileName + '_trackDistribution.root'),
    rpId_45_N = cms.uint32(3),
    rpId_45_F = cms.uint32(23),
    rpId_56_N = cms.uint32(103),
    rpId_56_F = cms.uint32(123),
)

process.ctppsProtonReconstructionPlotter = cms.EDAnalyzer("CTPPSProtonReconstructionPlotter",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"+alcarecoSuffix),
    tagRecoProtonsSingleRP = cms.InputTag("ctppsProtons"+alcarecoSuffix, "singleRP"),
    tagRecoProtonsMultiRP = cms.InputTag("ctppsProtons"+alcarecoSuffix, "multiRP"),
    outputFile = cms.string(options.outputFileName + '_protonReconstruction.root'),
    rpId_45_N = cms.uint32(3),
    rpId_45_F = cms.uint32(23),
    rpId_56_N = cms.uint32(103),
    rpId_56_F = cms.uint32(123),
)

if len(options.inputFiles) != 0:
    inputFiles = cms.untracked.vstring(options.inputFiles)
else:
    # Example input file
    inputFiles = cms.untracked.vstring(
        [
            "/store/data/Run2023C/AlCaPPSPrompt/ALCARECO/PPSCalMaxTracks-PromptReco-v3/000/367/696/00000/10d6bcd2-a5eb-4fd3-bc30-daeae548887e.root",
        ]
    )

print('Input files:\n',inputFiles, sep='')
process.source = cms.Source("PoolSource",
    fileNames = inputFiles,
    # Drop everything from the prompt alcareco besides the digis at input
    inputCommands = cms.untracked.vstring(
        'keep *'
    )
)

if options.jsonFileName != '':
    import FWCore.PythonUtilities.LumiList as LumiList
    jsonFileName = options.jsonFileName
    print("Using JSON file:",jsonFileName)
    process.source.lumisToProcess = LumiList.LumiList(filename = jsonFileName).getVLuminosityBlockRange()


# processing sequences
process.path = cms.Path(
    process.ctppsLHCInfoPlotter *
    process.ctppsTrackDistributionPlotter *
    process.ctppsProtonReconstructionPlotter
)
