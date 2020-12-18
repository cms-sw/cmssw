from __future__ import print_function
# Auto generated configuration file
# with command line options: stepALCA --datatier ALCARECO --conditions auto:run2_data -s ALCA:PromptCalibProdSiStripGains --eventcontent ALCARECO -n 1000 --dasquery=file dataset=/ZeroBias/Run2016C-SiStripCalMinBias-18Apr2017-v1/ALCARECO run=276243 --no_exec
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.register('era',"A", VarParsing.multiplicity.singleton, VarParsing.varType.string, "input era")
options.parseArguments()

import os

import Utilities.General.cmssw_das_client as das_client

###################################################################
def getFileNames_das_client(era_name):
###################################################################
    """Return files for given DAS query via das_client"""
    files = []

    query = "dataset dataset=/ZeroBias/Run2*"+era_name+"*SiStripCalMinBias-*/ALCARECO site=T2_CH_CERN"
    jsondict = das_client.get_data(query)
    status = jsondict['status']
    if status != 'ok':
        print("DAS query status: %s"%(status))
        return files

    data =  jsondict['data']
    viableDS = []
    for element in data:
        viableDS.append(element['dataset'][0]['name'])

    print("Using Dataset:",viableDS[-1])

    query = "file dataset=%s site=T2_CH_CERN | grep file.name" % viableDS[-1]
    jsondict = das_client.get_data(query)
    status = jsondict['status']
    if status != 'ok':
        print("DAS query status: %s"%(status))
        return files

    mongo_query = jsondict['mongo_query']
    filters = mongo_query['filters']
    data = jsondict['data']

    files = []
    for row in data:
        the_file = [r for r in das_client.get_value(row, filters['grep'])][0]
        if len(the_file) > 0 and not the_file in files:
            files.append(the_file)

    return files

###################################################################
process = cms.Process('testFromALCARECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')

###################################################################
# Messages
###################################################################
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiStripGainsPCLWorker=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    threshold = cms.untracked.string("DEBUG"),
    enableStatistics = cms.untracked.bool(True),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiStripGainsPCLWorker = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    )

process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.AlCaRecoStreams_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
    )


INPUTFILES=getFileNames_das_client(options.era)

if len(INPUTFILES)==0: 
    print("** WARNING: ** According to a DAS query no suitable data for test is available. Skipping test")
    os._exit(0)

myFiles = cms.untracked.vstring()
myFiles.extend([INPUTFILES[0][0].replace("\"","")])
# Input source
process.source = cms.Source("PoolSource",
                            fileNames = myFiles,
                            secondaryFileNames = cms.untracked.vstring()
                            )

process.options = cms.untracked.PSet()

# Additional output definition
process.ALCARECOStreamPromptCalibProdSiStripGains = cms.OutputModule("PoolOutputModule",
                                                                     SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiStripGains')),
                                                                     dataset = cms.untracked.PSet(dataTier = cms.untracked.string('ALCARECO'),
                                                                                                  filterName = cms.untracked.string('PromptCalibProdSiStripGains')),
                                                                     eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
                                                                     fileName = cms.untracked.string('PromptCalibProdSiStripGains_'+options.era+'.root'),
                                                                     outputCommands = cms.untracked.vstring('drop *', 
                                                                                                            'keep *_MEtoEDMConvertSiStripGains_*_*'
                                                                                                            )
                                                                     )

# Other statements
process.ALCARECOEventContent.outputCommands.extend(process.OutALCARECOPromptCalibProdSiStripGains_noDrop.outputCommands)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.ALCARECOStreamPromptCalibProdSiStripGainsOutPath = cms.EndPath(process.ALCARECOStreamPromptCalibProdSiStripGains)

# Schedule definition
process.schedule = cms.Schedule(process.pathALCARECOPromptCalibProdSiStripGains,process.endjob_step,process.ALCARECOStreamPromptCalibProdSiStripGainsOutPath)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

