from __future__ import print_function
import re,os
import FWCore.ParameterSet.Config as cms
from Configuration.DataProcessing.GetScenario import getScenario

"""
Example configuration for online reconstruction meant for visualization clients.
"""
from DQM.Integration.config.inputsource_cfi import options,runType,source

# this is needed to map the names of the run-types chosen by DQM to the scenarios, ideally we could converge to the same names
#scenarios = {'pp_run': 'ppEra_Run2_2016','cosmic_run':'cosmicsEra_Run2_2016','hi_run':'HeavyIons'}
#scenarios = {'pp_run': 'ppEra_Run2_2016','pp_run_stage1': 'ppEra_Run2_2016','cosmic_run':'cosmicsEra_Run2_2016','cosmic_run_stage1':'cosmicsEra_Run2_2016','hi_run':'HeavyIonsEra_Run2_HI'}
scenarios = {'pp_run': 'ppEra_Run2_2018','cosmic_run':'cosmicsEra_Run2_2018','hi_run':'ppEra_Run2_2018_pp_on_AA'}

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

# example of how to add a filer IN FRONT of all the paths, eg for HLT selection
#kwds['preFilter'] = 'DQM/Integration/python/config/visualizationPreFilter.hltfilter'

process = scenario.visualizationProcessing(writeTiers=['FEVT'], **kwds)

process.source = source
process.source.inputFileTransitionsEachEvent = cms.untracked.bool(True)
process.source.skipFirstLumis                = cms.untracked.bool(True)
process.source.minEventsPerLumi              = cms.untracked.int32(0)
process.source.nextLumiTimeoutMillis         = cms.untracked.int32(10000)
process.source.streamLabel                   = cms.untracked.string('streamDQMEventDisplay')

m = re.search(r"\((\w+)\)", str(source.runNumber))
runno = str(m.group(1))
outDir= '/fff/BU0/output/EvD/run'+runno+'/streamEvDOutput2'

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

process.DQMMonitoringService = cms.Service("DQMMonitoringService")

dump = False
if dump:
    psetFile = open("RunVisualizationProcessingCfg.py", "w")
    psetFile.write(process.dumpPython())
    psetFile.close()
    cmsRun = "cmsRun -e RunVisualizationProcessingCfg.py"
    print("Now do:\n%s" % cmsRun)
