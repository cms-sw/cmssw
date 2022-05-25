#!/usr/bin/env python3
"""
_pp_

Scenario supporting proton collisions

"""
from __future__ import print_function

import os
import sys

from Configuration.DataProcessing.Scenario import *
from Configuration.DataProcessing.Utils import stepALCAPRODUCER,stepSKIMPRODUCER,addMonitoring,dictIO,dqmIOSource,harvestingMode,dqmSeq,gtNameAndConnect
import FWCore.ParameterSet.Config as cms
from Configuration.DataProcessing.RecoTLR import customisePrompt,customiseExpress

class Reco(Scenario):
    def __init__(self):
        Scenario.__init__(self)
        self.recoSeq=''
        self.cbSc=self.__class__.__name__
        self.promptModifiers = cms.ModifierChain()
        self.expressModifiers = cms.ModifierChain()
        self.visModifiers = cms.ModifierChain()
    """
    _pp_

    Implement configuration building for data processing for proton
    collision data taking

    """


    def _checkRepackedFlag(self, options, **args):
        if 'repacked' in args:
            if args['repacked'] == True:
                options.isRepacked = True
            else:
                options.isRepacked = False
                


    def promptReco(self, globalTag, **args):
        """
        _promptReco_

        Proton collision data taking prompt reco

        """
        step = stepALCAPRODUCER(args['skims'])
        PhysicsSkimStep = ''
        if ("PhysicsSkims" in args) :
            PhysicsSkimStep = stepSKIMPRODUCER(args['PhysicsSkims'])
        dqmStep = dqmSeq(args,'')
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = self.cbSc
        if ('nThreads' in args) :
            options.nThreads=args['nThreads']

        miniAODStep = ''
        nanoAODStep = ''

        if 'outputs' in args:
            print(args['outputs']) 
            for a in args['outputs']:
                if a['dataTier'] == 'MINIAOD':
                    miniAODStep = ',PAT' 
                if a['dataTier'] in ['NANOAOD', 'NANOEDMAOD']:
                    nanoAODStep = ',NANO' 

        self._checkRepackedFlag(options, **args)

        if 'customs' in args:
            options.customisation_file=args['customs']

        eiStep=''

        options.step = 'RAW2DIGI,L1Reco,RECO'
        options.step += self.recoSeq + eiStep + step + PhysicsSkimStep
        options.step += miniAODStep + nanoAODStep
        options.step += ',DQM' + dqmStep + ',ENDJOB'

        dictIO(options,args)
        options.conditions = gtNameAndConnect(globalTag, args)
        
        process = cms.Process('RECO', cms.ModifierChain(self.eras, self.promptModifiers) )
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
        skims = args['skims']
        # the AlCaReco skims for PCL should only run during AlCaSkimming step which uses the same configuration on the Tier0 side, for this reason we drop them here
        pclWkflws = [x for x in skims if "PromptCalibProd" in x]
        for wfl in pclWkflws:
            skims.remove(wfl)
        
        step = stepALCAPRODUCER(skims)
        dqmStep= dqmSeq(args,'')
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = self.cbSc
        if ('nThreads' in args) :
            options.nThreads=args['nThreads']

        eiStep=''

        options.step = 'RAW2DIGI,L1Reco,RECO'+self.recoSeq+eiStep+step+',DQM'+dqmStep+',ENDJOB'

        dictIO(options,args)
        options.conditions = gtNameAndConnect(globalTag, args)

        
        options.filein = 'tobeoverwritten.xyz'
        if 'inputSource' in args:
            options.filetype = args['inputSource']
        process = cms.Process('RECO', cms.ModifierChain(self.eras, self.expressModifiers) )

        if 'customs' in args:
            options.customisation_file=args['customs']

        self._checkRepackedFlag(options,**args)

        cb = ConfigBuilder(options, process = process, with_output = True, with_input = True)

        cb.prepare()

        addMonitoring(process)
                
        return process


    def visualizationProcessing(self, globalTag, **args):
        """
        _visualizationProcessing_

        """

        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = self.cbSc
        # FIXME: do we need L1Reco here?
        options.step =''
        if 'preFilter' in args:
            options.step +='FILTER:'+args['preFilter']+','

        eiStep=''

        if 'beamSplashRun' in args:
            options.step += 'RAW2DIGI,L1Reco,RECO'+args['beamSplashRun']+',ENDJOB'
            print("Using RECO%s step in visualizationProcessing" % args['beamSplashRun'])
        else :
            options.step += 'RAW2DIGI,L1Reco,RECO'+eiStep+',ENDJOB'



        dictIO(options,args)
        options.conditions = gtNameAndConnect(globalTag, args)
        options.timeoutOutput = True
        # FIXME: maybe can go...maybe not
        options.filein = 'tobeoverwritten.xyz'

        if 'inputSource' in args:
            options.filetype = args['inputSource']
        else:
            # this is the default as this is what is needed on the OnlineCluster
            options.filetype = 'DQMDAQ'

        print("Using %s source"%options.filetype)            

        process = cms.Process('RECO', cms.ModifierChain(self.eras, self.visModifiers) )

        if 'customs' in args:
            options.customisation_file=args['customs']

        self._checkRepackedFlag(options, **args)

        cb = ConfigBuilder(options, process = process, with_output = True, with_input = True)

        cb.prepare()


        

        # FIXME: not sure abou this one...drop for the moment
        # addMonitoring(process)
                
        return process




    def alcaSkim(self, skims, **args):
        """
        _alcaSkim_

        AlcaReco processing & skims for proton collisions

        """

        step = ""
        pclWflws = [x for x in skims if "PromptCalibProd" in x]
        skims = [x for x in skims if x not in pclWflws]

        if len(pclWflws):
            step += 'ALCA:'+('+'.join(pclWflws))

        if len( skims ) > 0:
            if step != "":
                step += ","
            step += "ALCAOUTPUT:"+('+'.join(skims))

        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = self.cbSc
        options.step = step
        options.conditions = args['globaltag'] if 'globaltag' in args else 'None'
        if 'globalTagConnect' in args and args['globalTagConnect'] != '':
            options.conditions += ','+args['globalTagConnect']

        options.triggerResultsProcess = 'RECO'

        if 'customs' in args:
            options.customisation_file=args['customs']
        
        process = cms.Process('ALCA', self.eras)
        cb = ConfigBuilder(options, process = process)

        # Input source
        process.source = cms.Source(
           "PoolSource",
           fileNames = cms.untracked.vstring()
        )

        cb.prepare() 

        # FIXME: dirty hack..any way around this?
        # Tier0 needs the dataset used for ALCAHARVEST step to be a different data-tier
        for wfl in pclWflws:
            methodToCall = getattr(process, 'ALCARECOStream'+wfl)
            methodToCall.dataset.dataTier = cms.untracked.string('ALCAPROMPT')

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
        options.conditions = gtNameAndConnect(globalTag, args)
 
        process = cms.Process("HARVESTING", self.eras)
        process.source = dqmIOSource(args)

        if 'customs' in args:
            options.customisation_file=args['customs']

        configBuilder = ConfigBuilder(options, process = process)
        configBuilder.prepare()

        harvestingMode(process,datasetName,args,rANDl=False)
        return process


    def alcaHarvesting(self, globalTag, datasetName, **args):
        """
        _alcaHarvesting_

        Proton collisions data taking AlCa Harvesting

        """
        skims = []
        if 'skims' in args:
            skims = args['skims']


        if 'alcapromptdataset' in args:
            skims.append('@'+args['alcapromptdataset'])

        if len(skims) == 0: return None
        options = defaultOptions
        options.scenario = self.cbSc if hasattr(self,'cbSc') else self.__class__.__name__ 
        options.step = "ALCAHARVEST:"+('+'.join(skims))
        options.name = "ALCAHARVEST"
        options.conditions = gtNameAndConnect(globalTag, args)
 
        process = cms.Process("ALCAHARVEST", self.eras)
        process.source = cms.Source("PoolSource")

        if 'customs' in args:
            options.customisation_file=args['customs']

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
        options.conditions = gtNameAndConnect(globalTag, args)
        process = cms.Process("SKIM", self.eras)
        process.source = cms.Source("PoolSource")

        if 'customs' in args:
            options.customisation_file=args['customs']

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
        process = cms.Process('REPACK', self.eras)
        cb = ConfigBuilder(options, process = process, with_output = True,with_input=True)
        cb.prepare()
        print cb.pythonCfgCode
        return process
    """
