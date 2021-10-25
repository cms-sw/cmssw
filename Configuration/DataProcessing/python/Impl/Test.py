#!/usr/bin/env python3
"""
_Test_

Test Scenario implementation for unittests/development purposes

Not for use with data taking 

"""


from Configuration.DataProcessing.Scenario import Scenario
import FWCore.ParameterSet.Config as cms

class Test(Scenario):
    def __init__(self):
        Scenario.__init__(self)
    """
    _Test_

    Test Scenario

    """

    
    def promptReco(self, globalTag):
        """
        _promptReco_

        Returns skeleton process object

        """
        return cms.Process("RECO", self.eras)


    def expressProcessing(self, globalTag):
        """
        _expressProcessing_

        Returns skeleton process object

        """
        return cms.Process("Express", self.eras)


    def alcaSkim(self, skims):
        """
        _alcaSkim_

        Returns skeleton process object

        """
        return cms.Process("ALCARECO", self.eras)
        
        
    def dqmHarvesting(self, datasetName, runNumber,  globalTag, **args):
        """
        _dqmHarvesting_

        build a DQM Harvesting configuration

        this method can be used to test an extra scenario, all the 
        ConfigBuilder options can be overwritten by using **args. This will be
        useful for testing with real jobs.

        Arguments:
        
        datasetName - aka workflow name for DQMServer, this is the name of the
        dataset containing the harvested run
        runNumber - The run being harvested
        globalTag - The global tag being used
        inputFiles - The list of LFNs being harvested

        """
        options = defaultOptions
        options.scenario = "cosmics"
        options.step = "HARVESTING:dqmHarvesting"
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        options.name = "EDMtoMEConvert"
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        options.arguments = ""
        options.evt_type = ""
        options.filein = []

        options.__dict__.update(args)
 
        process = cms.Process("HARVESTING", self.eras)
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
        if 'saveByLumiSection' in args and \
                args.get('saveByLumiSection', ''):
            process.dqmSaver.saveByLumiSection = int(args['saveByLumiSection'])

        return process


    def skimming(self, *skims):
        """
        _skimming_

        Returns skeleton process object

        """
        return cms.Process("Skimming", self.eras)
