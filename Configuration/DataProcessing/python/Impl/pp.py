#!/usr/bin/env python
"""
_pp_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.PyReleaseValidation.ConfigBuilder import ConfigBuilder
from Configuration.PyReleaseValidation.ConfigBuilder import Options
from Configuration.PyReleaseValidation.ConfigBuilder import defaultOptions
from Configuration.DataProcessing.RecoTLR import customisePrompt,customiseExpress

class pp(Reco):
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
        skims = ['TkAlMinBias',
                 'TkAlMuonIsolated',
                 'MuAlCalIsolatedMu',
                 'MuAlOverlaps',
                 'HcalCalIsoTrk',
                 'HcalCalDijets',
                 'SiStripCalMinBias',
                 'EcalCalElectron',
                 'DtCalib',
                 'TkAlJpsiMuMu',
                 'TkAlUpsilonMuMu',
                 'TkAlZMuMu']

        process = self.promptRecoImp(self,globalTag, skims, args)

        #add the former top level patches here
        customisePrompt(process)
        
        return process


    def expressProcessing(self, globalTag, **args):
        """
        _expressProcessing_

        Proton collision data taking express processing

        """

        self.skims = ['SiStripCalZeroBias',
                      'TkAlMinBias',
                      'DtCalib',
                      'MuAlCalIsolatedMu',
                      'SiStripPCLHistos']
        process = self.expressProcessingImp(self, globalTag, skims, args)

        customiseExpress(process)
                
        return process


    def alcaHarvesting(self, globalTag, datasetName, **args):
        """
        _alcaHarvesting_

        Proton collisions data taking AlCa Harvesting

        """
        skims=['BeamSpotByRun'+
               'BeamSpotByLumi'+
               'SiStripQuality']
        
        options = defaultOptions
        options.scenario = "pp"
        options.step = "ALCAHARVEST:"+('+'.join(skims))
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

