## --
## Start with pre-defined skeleton process
## --
from PhysicsTools.PatAlgos.patTemplate_cfg import *
## ... and modify it according to the needs:
process.source.fileNames = [ '/store/data/Run2011A/SingleMu/AOD/PromptReco-v1/000/161/016/1C159A2D-D455-E011-9502-003048F01E88.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v1/000/161/016/1C1D1BBA-D255-E011-81F6-003048F1182E.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v1/000/161/016/2C9C9787-C755-E011-AC9A-0019DB2F3F9A.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v1/000/161/016/52713845-C855-E011-A1AB-000423D98B6C.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v1/000/161/016/702DDB20-E355-E011-BA84-0030487C2B86.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v1/000/161/016/78186858-D155-E011-B08A-003048F118C4.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v1/000/161/016/86102236-D455-E011-85C5-0030487CD812.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v1/000/161/016/9E612DE9-D455-E011-8FA6-003048F11C28.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v1/000/161/016/F8DFDD84-4B56-E011-84BD-003048D2BE08.root'
                           ]
# use the correct conditions
process.GlobalTag.globaltag = autoCond[ 'com10' ]
# use a sufficient number of events
process.maxEvents.input = 1000
# have a proper output file name
process.out.fileName = 'patTuple_triggerOnly_dataPromptReco.root'

## ---
## Define the path as empty skeleton
## ---
process.p = cms.Path(
)

### ===========
### PAT trigger
### ===========

## --
## Switch on
## --
from PhysicsTools.PatAlgos.tools.trigTools import switchOnTrigger
switchOnTrigger( process, sequence = 'p' ) # overwrite sequence default "patDefaultSequence", since it is not used in any path

## --
## Modify configuration according to your needs
## --
# add L1 algorithms' collection
process.patTrigger.addL1Algos     = cms.bool( True ) # default: 'False'
# add L1 objects collection-wise (latest collection)
process.patTrigger.l1ExtraMu      = cms.InputTag( 'l1extraParticles', ''            )
process.patTrigger.l1ExtraNoIsoEG = cms.InputTag( 'l1extraParticles', 'NonIsolated' )
process.patTrigger.l1ExtraIsoEG   = cms.InputTag( 'l1extraParticles', 'Isolated'    )
process.patTrigger.l1ExtraCenJet  = cms.InputTag( 'l1extraParticles', 'Central'     )
process.patTrigger.l1ExtraForJet  = cms.InputTag( 'l1extraParticles', 'Forward'     )
process.patTrigger.l1ExtraTauJet  = cms.InputTag( 'l1extraParticles', 'Tau'         )
process.patTrigger.l1ExtraETM     = cms.InputTag( 'l1extraParticles', 'MET'         )
process.patTrigger.l1ExtraHTM     = cms.InputTag( 'l1extraParticles', 'MHT'         )
# save references to original L1 objects
process.patTrigger.saveL1Refs = cms.bool( True ) # default: 'False'
# update event content to save
switchOnTrigger( process, sequence = 'p' )       # called once more to update the event content according to the changed parameters!!!
