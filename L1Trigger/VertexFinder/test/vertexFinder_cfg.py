#################################################################################################
# To run execute do
# cmsRun tmtt_tf_analysis_cfg.py Events=50 inputMC=Samples/Muons/PU0.txt histFile=outputHistFile.root
# where the arguments take default values if you don't specify them. You can change defaults below.
#################################################################################################

import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("L1TVertexFinder")

process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D17_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.load("FWCore.MessageLogger.MessageLogger_cfi")


options = VarParsing.VarParsing ('analysis')
options.register('analysis',False,VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool,"Run vertex finding analysis code")
options.register('histFile','Hist.root',VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,"Name of output histogram file")

options.parseArguments()


#--- input and output
inputFiles = []
for filePath in options.inputFiles:
    if filePath.endswith(".root"):
        inputFiles += filePath
    else:
        inputFiles += FileUtils.loadListFromFile(filePath)

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

if options.analysis:
    process.TFileService = cms.Service("TFileService", fileName = cms.string(options.histFile))

process.source = cms.Source ("PoolSource",
                            fileNames = cms.untracked.vstring(inputFiles),
                            secondaryFileNames = cms.untracked.vstring(),
                            # skipEvents = cms.untracked.uint32(500)
                            )


# process.out = cms.OutputModule("PoolOutputModule",
#     fileName = cms.untracked.string(options.outputFile),
#     outputCommands = cms.untracked.vstring(
#     	"keep *",
#     	"keep *_producer_*_*",
#     	"keep *_VertexProducer_*_*"
#     	)
# )


process.Timing = cms.Service("Timing", summaryOnly = cms.untracked.bool(True))


#--- Load config fragment that configures vertex producer
process.load('L1Trigger.VertexFinder.VertexProducer_cff')

#--- Load config fragment that configures vertex analyzer
process.load('L1Trigger.VertexFinder.VertexAnalyzer_cff')

if (options.analysis):
    process.p = cms.Path(process.VertexProducer + process.L1TVertexAnalyzer)
else:
    process.p = cms.Path(process.VertexProducer)

# process.e = cms.EndPath(process.out)
