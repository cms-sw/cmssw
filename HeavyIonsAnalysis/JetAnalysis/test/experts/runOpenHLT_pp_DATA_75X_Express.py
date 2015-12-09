import FWCore.ParameterSet.Config as cms
process = cms.Process('HiForest')
process.options = cms.untracked.PSet(
)

#parse command line arguments
from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('analysis')
options.register ('isPP',
                  False,
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.bool,
                  "Flag if this is a pp simulation")
options.parseArguments()

#####################################################################################
# HiForest labelling info
#####################################################################################


#####################################################################################
# Input source
#####################################################################################

process.source = cms.Source("PoolSource",
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
                           fileNames = cms.untracked.vstring(options.inputFiles[0])                            
)


# Number of events we want to process, -1 = all events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents))



#####################################################################################
# Load Global Tag, Geometry, etc.
#####################################################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
# set snapshot to future to allow centrality table payload.
process.GlobalTag.snapshotTime = cms.string("9999-12-31 23:59:59.000")


process.GlobalTag.toGet.extend([
 cms.PSet(record = cms.string("HeavyIonRcd"),
 connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
 tag = cms.string("CentralityTable_HFtowers200_Glauber2010A_eff99_run1v750x01_offline"),
 label = cms.untracked.string("HFtowers")
 ),
])



process.TFileService = cms.Service("TFileService",
                                  fileName=cms.string(options.outputFile))                                   


process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cff')


process.ana_step = cms.Path(
                            process.hltanalysis 
                            )
