#!/usr/bin/env python
"""
_AlCaTestEnable_

Scenario supporting proton collisions

"""

from Configuration.DataProcessing.Impl.AlCa import *

class AlCaTestEnable(AlCa):
    def __init__(self):
        AlCa.__init__(self)
        self.skims=['TkAlLAS']
    """
    _AlCaTestEnable_

    Implement configuration building for data processing for proton
    collision data taking

    """

    def expressProcessing(self, globalTag, **args):
        """
        _expressProcessing_

        Proton collision data taking express processing

        """
        skims = args['skims']

        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"
        options.step = stepALCAPRODUCER(skims)

        dictIO(options,args)
        options.conditions = gtNameAndConnect(globalTag, args)

        options.filein = 'tobeoverwritten.xyz'
        if 'inputSource' in args:
            options.filetype = args['inputSource']
        process = cms.Process('RECO', self.eras)

        if 'customs' in args:
            options.customisation_file=args['customs']

        cb = ConfigBuilder(options, process = process, with_output = True, with_input = True)

        cb.prepare()

        addMonitoring(process)

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
