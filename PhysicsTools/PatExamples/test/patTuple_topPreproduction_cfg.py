## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

## ---
## Adjust inputs if necessary
## ---
#from PhysicsTools.PatAlgos.tools.cmsswVersionTools import run36xOn35xInput
#run36xOn35xInput(process)

## --- 
## adjust workflow to need in TopPAG
## ---
from PhysicsTools.PatAlgos.tools.coreTools import *
removeCleaning(process)
removeMCMatching(process, ['All'])
removeSpecificPATObjects(process, ['Photons','Taus'])

## ---
## adjust content
## ---
process.patMuons.usePV      = False
process.patMuons.embedTrack = True

## define event content
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
process.out.outputCommands = cms.untracked.vstring('drop *', *patEventContentNoCleaning ) 
process.out.outputCommands+= [ 'keep edmTriggerResults_*_*_*',
                               'keep *_offlinePrimaryVertices_*_*'
                               ] 

process.p = cms.Path(
    process.patDefaultSequence
    )
