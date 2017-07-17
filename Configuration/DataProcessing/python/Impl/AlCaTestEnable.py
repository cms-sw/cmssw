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
        skims = []
        if 'skims' in args:
            skims = args['skims']
            if 'EcalTestPulsesRaw' not in args['skims']:
                skims.append('EcalTestPulsesRaw')
            pclWkflws = [x for x in skims if "PromptCalibProd" in x]
            for wfl in pclWkflws:
                skims.remove(wfl)

        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"
        options.step = stepALCAPRODUCER(skims)

        if 'outputs' in args:
            # the RAW data-tier needs a special treatment since the event-content as defined in release is not good enough
            outputs_Raw = [x for x in args['outputs'] if x['dataTier'] == 'RAW']
            outputs_noRaw = [x for x in args['outputs'] if x['dataTier'] != 'RAW']
            if len(outputs_Raw) == 1:
                print 'RAW data-tier requested'
            options.outputDefinition = outputs_noRaw.__str__()

        # dictIO(options,args)
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

        for output in outputs_Raw:
            print output
            moduleLabel = output['moduleLabel']
            selectEvents = output.get('selectEvents', None)
            maxSize = output.get('maxSize', None)

            outputModule = cms.OutputModule(
                "PoolOutputModule",
                fileName = cms.untracked.string("%s.root" % moduleLabel)
                )

            outputModule.dataset = cms.untracked.PSet(dataTier = cms.untracked.string("RAW"))

            if maxSize != None:
                outputModule.maxSize = cms.untracked.int32(maxSize)

            if selectEvents != None:
                outputModule.SelectEvents = cms.untracked.PSet(
                    SelectEvents = cms.vstring(selectEvents)
                    )
            outputModule.outputCommands = cms.untracked.vstring('drop *',
                                                                'keep  *_*_*_HLT')

            setattr(process, moduleLabel, outputModule)
	        # outputModule=getattr(self.process,theModuleLabel)
            setattr(process, moduleLabel+'_step', cms.EndPath(outputModule))
            path = getattr(process, moduleLabel+'_step')
            process.schedule.append(path)

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
