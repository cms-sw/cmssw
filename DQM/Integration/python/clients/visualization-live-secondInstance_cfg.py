import re,os
import FWCore.ParameterSet.Config as cms
from Configuration.DataProcessing.GetScenario import getScenario

"""
Example configuration for online reconstruction meant for visualization clients.
"""
from DQM.Integration.config.inputsource_cfi import options,runType,source

# this is needed to map the names of the run-types chosen by DQM to the scenarios, ideally we could converge to the same names
#scenarios = {'pp_run': 'ppRun2','cosmic_run':'cosmicsRun2','hi_run':'HeavyIons'}
scenarios = {'pp_run': 'ppRun2','pp_run_stage1': 'ppRun2','cosmic_run':'cosmicsRun2','cosmic_run_stage1':'cosmicsRun2','hi_run':'HeavyIons'}

if not runType.getRunTypeName() in scenarios.keys():
    msg = "Error getting the scenario out of the 'runkey', no mapping for: %s\n"%runType.getRunTypeName()
    raise RuntimeError, msg

scenarioName = scenarios[runType.getRunTypeName()]

print "Using scenario:",scenarioName


try:
    scenario = getScenario(scenarioName)
except Exception, ex:
    msg = "Error getting Scenario implementation for %s\n" % (
        scenarioName,)
    msg += str(ex)
    raise RuntimeError, msg


kwds = {}
# example of how to add a filer IN FRONT of all the paths, eg for HLT selection
#kwds['preFilter'] = 'DQM/Integration/python/config/visualizationPreFilter.hltfilter'

process = scenario.visualizationProcessing(globalTag='DUMMY', writeTiers=['FEVT'], **kwds)

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

process.load("DQM.Integration.config.FrontierCondition_GT_autoExpress_cfi")

process.options = cms.untracked.PSet(
        Rethrow = cms.untracked.vstring('ProductNotFound'),
        wantSummary = cms.untracked.bool(True)
    )

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
    )
oldo = process._Process__outputmodules["FEVToutput"]
del process._Process__outputmodules["FEVToutput"]

process.FEVToutput = cms.OutputModule("JsonWritingTimeoutPoolOutputModule",
    splitLevel = oldo.splitLevel,
    eventAutoFlushCompressedSize = oldo.eventAutoFlushCompressedSize,
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
    print "Now do:\n%s" % cmsRun
