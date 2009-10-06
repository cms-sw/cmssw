#!/usr/bin/env python
"""
_Cosmics_

Scenario supporting cosmic data taking

"""

import os
import sys

from Configuration.DataProcessing.Scenario import Scenario
import FWCore.ParameterSet.Config as cms
from Configuration.PyReleaseValidation.ConfigBuilder import ConfigBuilder
from Configuration.PyReleaseValidation.ConfigBuilder import Options
from Configuration.PyReleaseValidation.ConfigBuilder import defaultOptions
from Configuration.PyReleaseValidation.ConfigBuilder import installFilteredStream
    


class Cosmics(Scenario):
    """
    _Cosmics_

    Implement configuration building for data processing for cosmic
    data taking

    """


    def promptReco(self, globalTag, writeTiers = ['RECO']):
        """
        _promptReco_

        Cosmic data taking prompt reco

        """
        
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "cosmics"
        options.step = 'RAW2DIGI,RECO,DQM,ENDJOB'
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        options.magField = 'AutoFromDBCurrent'
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag

        
        
        process = cms.Process('RECO')
        cb = ConfigBuilder(options, process = process)

        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring()
        )
        cb.prepare()

        for tier in writeTiers: 
          addOutputModule(process, tier, process.RECOEventContent)        
 
        return process

    def expressProcessing(self, globalTag,  writeTiers = [],
                          datasets = [], alcaDataset = None):
        """
        _expressProcessing_

        Implement Cosmics Express processing

        Based on/Edited from:
        
        ConfigBuilder.py
             step2
             -s RAW2DIGI,RECO:reconstructionCosmics,ALCA:MuAlCalIsolatedMu\
             +RpcCalHLT+TkAlCosmicsHLT+TkAlCosmics0T\
             +MuAlStandAloneCosmics+MuAlGlobalCosmics\
             +HcalCalHOCosmics
             --datatier RECO
             --eventcontent RECO
             --conditions FrontierConditions_GlobalTag,CRAFT_30X::All
             --scenario cosmics
             --no_exec
             --data        
        
        """

        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "cosmics"
        options.step = \
          """RAW2DIGI,RECO:reconstructionCosmics,ALCA:MuAlCalIsolatedMu+RpcCalHLT+TkAlCosmicsHLT+TkAlCosmics0T+MuAlStandAloneCosmics+MuAlGlobalCosmics+HcalCalHOCosmics"""
        options.isMC = False
        options.isData = True
        options.eventcontent = "RECO"
        options.relval = None
        options.beamspot = None
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        
        process = cms.Process('EXPRESS')
        cb = ConfigBuilder(options, process = process)
        process.load(cb.EVTCONTDefaultCFF)

        #  //
        # // Install the OutputModules for everything but ALCA
        #//
        self.addExpressOutputModules(process, writeTiers, datasets)
        
        #  //
        # // TODO: Install Alca output
        #//
        
        process.source = cms.Source(
            "NewEventStreamFileReader",
            fileNames = cms.untracked.vstring()
        )
        cb.prepare()        
        
        return process
    

    def alcaReco(self, *skims):
        """
        _alcaReco_

        AlcaReco processing & skims for cosmics

        Based on:
        Revision: 1.120 
        ConfigBuilder.py 
          step3_V16
          -s ALCA:MuAlStandAloneCosmics+DQM
          --scenario cosmics
          --conditions FrontierConditions_GlobalTag,CRAFT_V16P::All
          --no_exec --data


        Expecting GlobalTag to be provided via API initially although
        this may not be the case

        """
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "cosmics"
        options.step = 'ALCA:MuAlStandAloneCosmics+DQM'
        options.isMC = False
        options.isData = True
        options.conditions = "FrontierConditions_GlobalTag,CRAFT_V16P::All" 
        options.beamspot = None
        options.eventcontent = None
        options.relval = None
        

        
        process = cms.Process('ALCA')
        cb = ConfigBuilder(options, process = process)
        cb.addStandardSequences()
        cb.addConditions()
        process.load(cb.EVTCONTDefaultCFF)
        # import of standard configurations
        process.load('Configuration/StandardSequences/Services_cff')
        process.load('FWCore/MessageService/MessageLogger_cfi')
        process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
        process.load('Configuration/StandardSequences/GeometryIdeal_cff')
        process.load('Configuration/StandardSequences/MagneticField_38T_cff')
        process.load('Configuration/StandardSequences/AlCaRecoStreams_cff')
        process.load('Configuration/StandardSequences/EndOfProcess_cff')
        process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
        process.load('Configuration/EventContent/EventContentCosmics_cff')
        
        process.configurationMetadata = cms.untracked.PSet(
            version = cms.untracked.string('$Revision: 1.11 $'),
            annotation = cms.untracked.string('step3_V16 nevts:1'),
            name = cms.untracked.string('PyReleaseValidation')
        )
        process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(1)
        )
        process.options = cms.untracked.PSet(
            Rethrow = cms.untracked.vstring('ProductNotFound')
        )
        # Input source
        process.source = cms.Source(
            "PoolSource",
            fileNames = cms.untracked.vstring()
        )
        
        # Additional output definition
        process.ALCARECOStreamMuAlStandAloneCosmics = cms.OutputModule("PoolOutputModule",
            SelectEvents = cms.untracked.PSet(
                SelectEvents = cms.vstring('pathALCARECOMuAlStandAloneCosmics')
            ),
            outputCommands = cms.untracked.vstring('drop *', 
                'keep *_ALCARECOMuAlStandAloneCosmics_*_*', 
                'keep *_muonCSCDigis_*_*', 
                'keep *_muonDTDigis_*_*', 
                'keep *_muonRPCDigis_*_*', 
                'keep *_dt1DRecHits_*_*', 
                'keep *_dt2DSegments_*_*', 
                'keep *_dt4DSegments_*_*', 
                'keep *_csc2DRecHits_*_*', 
                'keep *_cscSegments_*_*', 
                'keep *_rpcRecHits_*_*'),
            fileName = cms.untracked.string('ALCARECOMuAlStandAloneCosmics.root'),
            dataset = cms.untracked.PSet(
                filterName = cms.untracked.string('StreamALCARECOMuAlStandAloneCosmics'),
                dataTier = cms.untracked.string('ALCARECO')
            )
        )
        
        
        #  //
        # // Verify and Edit the list of skims to be written out
        #//  by this job
        availableStreams = process.outputModules_().keys()

        #  //
        # // First up: Verify skims are available by output module name
        #//
        for skim in skims:
            if skim not in availableStreams:
                msg = "Skim named: %s not available " % skim
                msg += "in Alca Reco Config:\n"
                msg += "Known Skims: %s\n" % availableStreams
                raise RuntimeError, msg

        #  //
        # // Prune any undesired skims
        #//
        for availSkim in availableStreams:
            if availSkim not in skims:
                self.dropOutputModule(process, availSkim)

        return process
                

        

        


    def dqmHarvesting(self, datasetName, runNumber,  globalTag, **options):
        """
        _dqmHarvesting_

        Cosmic data taking DQM Harvesting

        """
        options = defaultOptions
        options.scenario = "cosmics"
        options.step = "HARVESTING:dqmHarvesting"
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        options.name = "EDMtoMEConvert"
        options.number = -1
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        options.arguments = ""
        options.evt_type = ""
        options.filein = []
        options.gflash = False
        options.customisation_file = ""

 
        process = cms.Process("HARVESTING")
        process.source = cms.Source("PoolSource")
        configBuilder = ConfigBuilder(options, process = process)
        configBuilder.prepare()

        #
        # customise process for particular job
        #
        process.source.processingMode = cms.untracked.string('RunsAndLumis')
        process.source.fileNames = cms.untracked(cms.vstring())
        process.maxEvents.input = -1
        process.dqmSaver.workflow = datasetName
        
        return process
