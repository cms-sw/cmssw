## --
## Start with pre-defined skeleton process
## --
from PhysicsTools.PatAlgos.patTemplate_cfg import *
## ... and modify it according to the needs:
# use latest ReReco as input
from PhysicsTools.PatExamples.samplesCERN_cff import dataMu
process.source.fileNames = dataMu
# use the correct conditions
process.GlobalTag.globaltag = 'GR_R_38X_V15::All'
# use a sufficient number of events
process.maxEvents.input = 1000
# have a proper output file name
process.out.fileName = 'patTuple_triggerOnly_dataReReco.root'

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
