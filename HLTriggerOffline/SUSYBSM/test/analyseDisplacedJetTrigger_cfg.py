#===============================================================
# STUDY EXOTICA DISPLACED JET TRIGGER
# Documented in header of AnalyseDisplacedJetTrigger.h
#===============================================================

#######################################################################
# Specify if data/MC and name of process used when producing HLT info #
#######################################################################
DATA = False
hltTriggerProcess = "HLT"

# Only analyse events passing this control trigger, chosen to selected an unbiased event sample for the
# displaced jet trigger. If empty, then all events are used.
skimTriggers =  ['HLT_HT240_v1']
#skimTriggers =  [] 

#########################
# Convert to PAT format #
#########################

## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *
from PhysicsTools.PatAlgos.tools.coreTools import *

if DATA:
  print "Reading real data"
  process.GlobalTag.globaltag = 'GR_R_311_V1::All'
#  Don't use MC truth if real data ...
  runOnData(process)

else:
  print "Reading MC"
  process.GlobalTag.globaltag = 'START311_V1::All'

# Kill all PAT objects except jets.
removeAllPATObjectsBut(process, ['Jets','METs'] )

# Apply cut to PAT jets.
process.selectedPatJets.cut = cms.string("et > 30 && abs(eta) < 3.0")

# Enable exotica displaced jet tagger.
process.patJets.trackAssociationSource = cms.InputTag("displacedJetAssociator")
process.patJets.discriminatorSources = cms.VInputTag(
        cms.InputTag("displacedJetTags"),
)

# Disable cleaning
removeCleaning(process)

# enable PAT trigger functionality
from PhysicsTools.PatAlgos.tools.trigTools import *
switchOnTrigger(process)

# Specify which trigger table to use
process.patTrigger.processName = hltTriggerProcess
process.patTriggerEvent.processName = hltTriggerProcess
print "Using trigger menu ", process.patTriggerEvent.processName

####################################################
# Run algorithm to tag jets with few prompt tracks #
####################################################

process.load("HLTriggerOffline.SUSYBSM.displacedJetTagger_cff")

#########################
# Specify input dataset #
#########################
                                         
process.source.fileNames = [          
    'file:/opt/ppd/cms/users/tomalin/sim_reco.root'

#    'file:/opt/ppd/cms/users/tomalin/data_reco1.root',
]

process.maxEvents.input = -1

process.outpath.remove(process.out)

process.MessageLogger.cerr.INFO.limit = 10
#process.MessageLogger.categories = cms.untracked.vstring('AnalyseDisplacedJetTrigger')

process.options.SkipEvent = cms.untracked.vstring('ProductNotFound')

################################
# Set up DQM for histogramming #
################################

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.DQMStore.verbose = 1
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow=cms.untracked.string("/HLT/SUSYBSM/Validation")
# Either save a new ROOT histogram file at the end of each run ...
#process.dqmSaver.saveByRun = 1
#process.dqmSaver.saveAtJobEnd = False
# Or save a new ROOT histogram file only at the end of the job ...
process.dqmSaver.saveByRun = -1
process.dqmSaver.forceRunNumber = 1
process.dqmSaver.saveAtJobEnd = True

process.ep = cms.EndPath(process.dqmSaver)

import HLTrigger.HLTfilters.hltHighLevel_cfi
process.skimUsingHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
       andOr = True, # accept OR of triggers.
       HLTPaths = skimTriggers,
       TriggerResultsTag = cms.InputTag("TriggerResults","",hltTriggerProcess)
)
                                         
#######################################
# Define options for trigger analysis #
#######################################

process.load("HLTriggerOffline.SUSYBSM.analyseDisplacedJetTrigger_cff")

####################################################################################
# Optionally use only events passing a trigger giving an unbiased sample of events #
####################################################################################

if len(skimTriggers) > 0:

  process.p = cms.Path(
      process.skimUsingHLT * process.displacedJetSequence * process.patDefaultSequence * process.analysis  
  )

else:

  process.p = cms.Path(
      process.displacedJetSequence * process.patDefaultSequence * process.analysis 
  )

# Kill PF tau path, which seems present for no good reason ...
process.p0 = cms.Path()



