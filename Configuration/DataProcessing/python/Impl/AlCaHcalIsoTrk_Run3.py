#!/usr/bin/env python3
"""
_AlCaHcalIsoTrk_Run3_

Scenario supporting proton collisions for AlCa needs for AlCaHcalIsoTrk data stream

"""

from Configuration.DataProcessing.Scenario import *
from Configuration.DataProcessing.Utils import stepALCAPRODUCER,dictIO,gtNameAndConnect
from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.DataProcessing.Impl.pp import pp
import FWCore.ParameterSet.Config as cms

class AlCaHcalIsoTrk_Run3(pp):
    def __init__(self):
        Scenario.__init__(self)
        self.eras=Run3
        self.skims=["HcalCalIsoTrkFromAlCaRaw"]

    """
    _AlCaHcalIsoTrk_Run3_

    Implement configuration building for data processing for proton
    collision data taking AlCaHcalIsoTrk AlCaRaw

    """

    def promptReco(self, globalTag, **args):
        """
        _promptReco_

        Proton collision data taking prompt reco

        """
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"

        if ('nThreads' in args):
            options.nThreads=args['nThreads']

        options.step = stepALCAPRODUCER(args['skims'])

        dictIO(options,args)
        options.conditions = gtNameAndConnect(globalTag, args)
        
        process = cms.Process('RECO', cms.ModifierChain(self.eras))
        cb = ConfigBuilder(options, process = process, with_output = True)

        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring()
        )
        cb.prepare()

        return process
