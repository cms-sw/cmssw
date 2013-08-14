# This is an example PAT configuration showing the usage of PAT on full sim samples

# Starting with a skeleton process which gets imported with the following line
from PhysicsTools.PatAlgos.patTemplate_cfg import *

# ----------------------------------------------------
# EXAMPLE 1: restrict input to AOD
# ----------------------------------------------------
#from PhysicsTools.PatAlgos.tools.coreTools import *
#restrictInputToAOD(process)

# ----------------------------------------------------
# EXAMPLE 2: remove MC matching from PAT default
#            sequences
# ----------------------------------------------------
#from PhysicsTools.PatAlgos.tools.coreTools import *
#removeMCMatching(process, ['All'])

# ----------------------------------------------------
# EXAMPLE 3: remove certain object collections from
#            the PAT workflow
# ----------------------------------------------------
#from PhysicsTools.PatAlgos.tools.coreTools import *
#removeAllPATObjectsBut(process, ['Muons'])
#removeSpecificPATObjects(process, ['Electrons', 'Muons', 'Taus'])

# let it run
process.p = cms.Path(
    process.patDefaultSequence
    )

# ----------------------------------------------------
# You might want to change some of these default
# parameters
# ----------------------------------------------------
#process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#process.source.fileNames = [
#'/store/relval/CMSSW_3_1_1/RelValCosmics/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/7625DA7D-E36B-DE11-865A-000423D174FE.root'
#                            ]         ##  (e.g. 'file:AOD.root')
#process.maxEvents.input = ...         ##  (e.g. -1 to run on all events)
#process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#process.out.fileName = ...            ##  (e.g. 'myTuple.root')
#process.options.wantSummary = True    ##  (to suppress the long output at the end of the job)    
