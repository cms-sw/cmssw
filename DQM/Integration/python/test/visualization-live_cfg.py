from Configuration.DataProcessing.GetScenario import getScenario

"""
Example configuration for online reconstruction meant for visualization clients.
"""


from DQM.Integration.test.inputsource_cfi import options,runType,source





# this is needed to map the names of the run-types chosen by DQM to the scenarios, ideally we could converge to the same names
scenarios = {'pp_run': 'ppRun2','cosmic_run':'cosmicsRun2','hi_run':'HeavyIons'}


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
#kwds['preFilter'] = 'DQM/Integration/python/test/visualizationPreFilter.hltfilter'

process = scenario.visualizationProcessing(globalTag='DUMMY', writeTiers=['FEVT'], **kwds)

process.source = source

process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

process.options = cms.untracked.PSet(
        Rethrow = cms.untracked.vstring('ProductNotFound'),
        wantSummary = cms.untracked.bool(True)
    )

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
    )



dump = False
if dump:
    psetFile = open("RunVisualizationProcessingCfg.py", "w")
    psetFile.write(process.dumpPython())
    psetFile.close()
    cmsRun = "cmsRun -e RunVisualizationProcessingCfg.py"
    print "Now do:\n%s" % cmsRun
