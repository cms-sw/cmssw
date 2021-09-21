from __future__ import print_function
import re, os, sys
import FWCore.ParameterSet.Config as cms
from Configuration.DataProcessing.GetScenario import getScenario

"""
Example configuration for online reconstruction meant for visualization clients.
"""

unitTest = False
if 'unitTest=True' in sys.argv:
    unitTest=True

if unitTest:
    from DQM.Integration.config.unittestinputsource_cfi import options, runType, source
else:
    from DQM.Integration.config.inputsource_cfi import options, runType, source

# this is needed to map the names of the run-types chosen by DQM to the scenarios, ideally we could converge to the same names
#scenarios = {'pp_run': 'ppEra_Run2_2016','cosmic_run':'cosmicsEra_Run2_2016','hi_run':'HeavyIons'}
#scenarios = {'pp_run': 'ppEra_Run2_2016','pp_run_stage1': 'ppEra_Run2_2016','cosmic_run':'cosmicsEra_Run2_2016','cosmic_run_stage1':'cosmicsEra_Run2_2016','hi_run':'HeavyIonsEra_Run2_HI'}
scenarios = {'pp_run': 'ppEra_Run3','cosmic_run':'cosmicsEra_Run3','hi_run':'ppEra_Run2_2016_pA'}

if not runType.getRunTypeName() in scenarios.keys():
    msg = "Error getting the scenario out of the 'runkey', no mapping for: %s\n"%runType.getRunTypeName()
    raise RuntimeError(msg)

scenarioName = scenarios[runType.getRunTypeName()]

print("Using scenario:",scenarioName)

try:
    scenario = getScenario(scenarioName)
except Exception as ex:
    msg = "Error getting Scenario implementation for %s\n" % (
        scenarioName,)
    msg += str(ex)
    raise RuntimeError(msg)

# A hack necessary to prevert scenario.visualizationProcessing
# from overriding the connect string
from DQM.Integration.config.FrontierCondition_GT_autoExpress_cfi import GlobalTag
kwds = {
   'globalTag': GlobalTag.globaltag.value(),
   'globalTagConnect': GlobalTag.connect.value()
}

# explicitly select the input collection, since we get multiple in online
from EventFilter.RawDataCollector.rawDataMapperByLabel_cfi import rawDataMapperByLabel
rawDataMapperByLabel.rawCollectionList = ["rawDataRepacker"]

# example of how to add a filer IN FRONT of all the paths, eg for HLT selection
#kwds['preFilter'] = 'DQM/Integration/config/visualizationPreFilter.hltfilter'

# The following filter was used during 2018 high pile up (HPU) run.
#kwds['preFilter'] = 'DQM/Integration/config/visualizationPreFilter.pixelClusterFilter'

process = scenario.visualizationProcessing(writeTiers=['FEVT'], **kwds)

if unitTest:
    process.__dict__['_Process__name'] = "RECONEW"

process.source = source

if not unitTest:
    process.source.inputFileTransitionsEachEvent = True
    process.source.skipFirstLumis                = True
    process.source.minEventsPerLumi              = 0
    process.source.nextLumiTimeoutMillis         = 10000
    process.source.streamLabel                   = 'streamDQMEventDisplay'

    m = re.search(r"\((\w+)\)", str(source.runNumber))
    runno = str(m.group(1))
    outDir= '/fff/BU0/output/EvD/run'+runno+'/streamEvDOutput2'
else:
    runno = options.runNumber
    outDir = "./upload"

#create output directory
try:
    os.makedirs(outDir)
except:
    pass

process.options = cms.untracked.PSet(
        Rethrow = cms.untracked.vstring('ProductNotFound'),
        wantSummary = cms.untracked.bool(True),
        numberOfThreads = cms.untracked.uint32(8),
        numberOfStreams = cms.untracked.uint32(8)
    )

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
    )
oldo = process._Process__outputmodules["FEVToutput"]
del process._Process__outputmodules["FEVToutput"]

process.FEVToutput = cms.OutputModule("JsonWritingTimeoutPoolOutputModule",
    splitLevel = oldo.splitLevel,
    outputCommands = oldo.outputCommands,
    fileName = oldo.fileName,
    dataset = oldo.dataset,
    runNumber = cms.untracked.uint32(int(runno)),
    streamLabel = cms.untracked.string("streamEvDOutput2_dqmcluster"),
    # output path must exist!
    outputPath = cms.untracked.string(outDir),
)

if hasattr(oldo, 'SelectEvents'):
    process.FEVToutput.SelectEvents = oldo.SelectEvents

process.DQMMonitoringService = cms.Service("DQMMonitoringService")

dump = False
if dump:
    psetFile = open("RunVisualizationProcessingCfg.py", "w")
    psetFile.write(process.dumpPython())
    psetFile.close()
    cmsRun = "cmsRun -e RunVisualizationProcessingCfg.py"
    print("Now do:\n%s" % cmsRun)
