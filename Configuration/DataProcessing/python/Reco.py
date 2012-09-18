#!/usr/bin/env python
"""
_pp_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Scenario import *
from Configuration.DataProcessing.Utils import stepALCAPRODUCER,addMonitoring,dictIO,dqmIOSource,harvestingMode,dqmSeq
import FWCore.ParameterSet.Config as cms
from Configuration.DataProcessing.RecoTLR import customisePrompt,customiseExpress

class Reco(Scenario):
    def __init__(self):
        self.recoSeq=''
        self.cbSc=self.__class__.__name__
    """
    _pp_

    Implement configuration building for data processing for proton
    collision data taking

    """

    def promptReco(self, globalTag, **args):
        """
        _promptReco_

        Proton collision data taking prompt reco

        """
        step = stepALCAPRODUCER(args['skims'])
        dqmStep= dqmSeq(args,'')
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = self.cbSc
        options.step = 'RAW2DIGI,L1Reco,RECO'+self.recoSeq+step+',DQM'+dqmStep+',ENDJOB'
        dictIO(options,args)
        options.conditions = globalTag
        
        process = cms.Process('RECO')
        cb = ConfigBuilder(options, process = process, with_output = True)

        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring()
        )
        cb.prepare()

        addMonitoring(process)
        
        return process


    def expressProcessing(self, globalTag, **args):
        """
        _expressProcessing_

        Proton collision data taking express processing

        """
        step = stepALCAPRODUCER(args['skims'])
        dqmStep= dqmSeq(args,'')
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = self.cbSc
        options.step = 'RAW2DIGI,L1Reco,RECO'+step+',DQM'+dqmStep+',ENDJOB'
        dictIO(options,args)
        options.conditions = globalTag
        
        process = cms.Process('RECO')
        cb = ConfigBuilder(options, process = process, with_output = True)

        # Input source
        process.source = cms.Source("NewEventStreamFileReader",
            fileNames = cms.untracked.vstring()
        )
        cb.prepare()

        addMonitoring(process)
                
        return process


    def alcaSkim(self, skims, **args):
        """
        _alcaSkim_

        AlcaReco processing & skims for proton collisions

        """

        step = ""
        if 'PromptCalibProd' in skims:
            step = "ALCA:PromptCalibProd" 
            skims.remove('PromptCalibProd')
        
        if len( skims ) > 0:
            if step != "":
                step += ","
            step += "ALCAOUTPUT:"+('+'.join(skims))
                
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = self.cbSc
        options.step = step
        options.conditions = args['globaltag'] if 'globaltag' in args else 'None'
        options.triggerResultsProcess = 'RECO'
        
        process = cms.Process('ALCA')
        cb = ConfigBuilder(options, process = process)

        # Input source
        process.source = cms.Source(
           "PoolSource",
           fileNames = cms.untracked.vstring()
        )

        cb.prepare() 

        # FIXME: dirty hack..any way around this?
        # Tier0 needs the dataset used for ALCAHARVEST step to be a different data-tier
        if 'PromptCalibProd' in step:
            process.ALCARECOStreamPromptCalibProd.dataset.dataTier = cms.untracked.string('ALCAPROMPT')

        return process


    def dqmHarvesting(self, datasetName, runNumber, globalTag, **args):
        """
        _dqmHarvesting_

        Proton collisions data taking DQM Harvesting

        """
        options = defaultOptions
        options.scenario = self.cbSc
        options.step = "HARVESTING"+dqmSeq(args,':dqmHarvesting')
        options.name = "EDMtoMEConvert"
        options.conditions = globalTag
 
        process = cms.Process("HARVESTING")
        process.source = dqmIOSource(args)
        configBuilder = ConfigBuilder(options, process = process)
        configBuilder.prepare()

        harvestingMode(process,datasetName,args,rANDl=False)
        return process


    def alcaHarvesting(self, globalTag, datasetName, **args):
        """
        _alcaHarvesting_

        Proton collisions data taking AlCa Harvesting

        """
        if not 'skims' in args: return None
        options = defaultOptions
        options.scenario = self.cbSc if hasattr(self,'cbSc') else self.__class__.__name__ 
        options.step = "ALCAHARVEST:"+('+'.join(args['skims']))
        options.name = "ALCAHARVEST"
        options.conditions = globalTag
 
        process = cms.Process("ALCAHARVEST")
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

    def skimming(self, skims, globalTag,**options):
        """
        _skimming_

        skimming method overload for the prompt skiming
        
        """
        options = defaultOptions
        options.scenario = self.cbSc if hasattr(self,'cbSc') else self.__class__.__name__
        options.step = "SKIM:"+('+'.join(skims))
        options.name = "SKIM"
        options.conditions = globalTag
        process = cms.Process("SKIM")
        process.source = cms.Source("PoolSource")
        configBuilder = ConfigBuilder(options, process = process)
        configBuilder.prepare()

        return process
        
    """
    def repack(self, **args):
        options = defaultOptions
        dictIO(options,args)
        options.filein='file.dat'
        options.filetype='DAT'
        options.scenario = self.cbSc if hasattr(self,'cbSc') else self.__class__.__name__
        process = cms.Process('REPACK')
        cb = ConfigBuilder(options, process = process, with_output = True,with_input=True)
        cb.prepare()
        print cb.pythonCfgCode
        return process
    """
