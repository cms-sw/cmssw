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
from Configuration.PyReleaseValidation.ConfigBuilder import defaultOptions
from Configuration.PyReleaseValidation.ConfigBuilder import installFilteredStream
    


class Cosmics(Scenario):
    """
    _Cosmics_

    Implement configuration building for data processing for cosmic
    data taking

    """


    def promptReco(self,process, recoOutputModule, aodOutputModule = None,
                   alcaOutputModule = None):
        """
        _promptReco_

        Cosmic data taking prompt reco

        """
        
        options = defaultOptions
        options.scenario = "cosmics"
        options.step = 'RAW2DIGI,RECO'
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        
        cb = ConfigBuilder(options, process = process)
        cb.addStandardSequences()
        cb.addConditions()
        process.load(cb.EVTCONTDefault)
        recoOutputModule.eventContent = process.RECOEventContent
        if aodOutputModule != None:
            aodOutputModule.eventContent = process.AODEventContent
        if alcaOutputModule != None:
            print "ALCA Output needs to be implemented"
            #alcaOutputModule.eventContent = \
            #     process.ALCAIntermediateEventContent
        return process

    def alcaReco(self, process, *skims):
        """
        _alcaReco_

        AlcaReco processing & skims for cosmics

        """
        # map skim name to args to installFilteredStream
        possibleSkims = {
            "ALCARECOStreamHcalCalHOCosmics" : (
            process.schedule,
            "Configuration/StandardSequences/AlCaRecoStreams_cff"
            ),

            "ALCARECOStreamTkAlCosmics": (
            process.schedule,
            "Configuration/StandardSequences/AlCaRecoStreams_cff"
            ),
            "ALCARECOStreamTkAlCosmicsHLT" : (
             process.schedule,
            "Configuration/StandardSequences/AlCaRecoStreams_cff"
            ),
            "ALCARECOStreamMuAlStandAloneCosmics": (
             process.schedule,
            "Configuration/StandardSequences/AlCaRecoStreams_cff"
            ),
            "ALCARECOStreamMuAlGlobalCosmics": (
             process.schedule,
            "Configuration/StandardSequences/AlCaRecoStreams_cff"
            ),
            
            }
        skimname = "ALCA:"
        for skim in skims:
            if skim not in possibleSkims.keys():
                msg = "Tried to install unknown skim: %s\n" % skim
                msg += "Known skims for scenario %s are:\n" % (
                    self.__class__.__name__)
                for pskim in possibleSkims.keys():
                    msg += "  => %s\n" % pskim
                raise RuntimeError, msg
            skimname += "%s+" % skim
        if skimname.endswith("+"):
            skimname = skimname[:-1]

        options = defaultOptions
        options.scenario = "cosmics"
        options.step = skimname
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        cb = ConfigBuilder(options, process = process)
        #cb._options.step = 'ALCA' # no idea if this works????
        cb.addStandardSequences()
        cb.addConditions()
        process.load(cb.EVTCONTDefault)

        for skim in skims:
            installFilteredStream(process, possibleSkims[skim][0],
                                  skim, possibleSkims[skim][1])
            
        
        return process
        


    def dqmHarvesting(self, datasetName, runNumber,  globalTag,
                      *inputFiles, **options):
        """
        _dqmHarvesting_

        Cosmic data taking DQM Harvesting

        """

        #
        # 
        #
        #sys.argv=["cmsDriver.py",
        #          "step3",
        #          "-s", "HARVESTING:dqmHarvesting",
        #          "--conditions", "FrontierConditions_GlobalTag,CRAFT_30X::All",
        #          "--filein", "file:step2_RAW2DIGI_RECO_DQM.root",
        #          "--data",
        #          "--scenario=cosmics"]
        #  //
        # // should be able to translate these into options settngs
        #//
        options = defaultOptions
        options.scenario = "cosmics"
        options.step = "HARVESTING:dqmHarvesting"
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        options.name = "EDMtoMEConvert"
        options.number = -1
        options.conditions = "FrontierConditions_GlobalTag,CRAFT_30X::All"
        options.filein = list(inputFiles)
        options.arguments = ""
        options.evt_type = ""
        options.customisation_file = ""
        configBuilder = ConfigBuilder(options)


        configBuilder.prepare()
        process = configBuilder.process

        #
        # customise process for particular job
        #
        process.source.fileNames = cms.untracked(cms.vstring())
        #for fileName in inputFiles:
        #    process.source.fileNames.append(fileName)
        
        process.maxEvents.input = -1
        process.dqmSaver.workflow = datasetName
        
        return process
