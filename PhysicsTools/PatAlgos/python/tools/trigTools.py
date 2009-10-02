import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.patEventContent_cff import *
    
def switchOnTrigger( process ):
    """ Enables trigger information in PAT  """
    ## add trigger modules to path
    process.load("PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cff")
    process.patDefaultSequence += process.patTriggerSequence

    ## configure pat trigger
    process.patTrigger.onlyStandAlone = False

    ## add trigger specific event content to PAT event content
    process.out.outputCommands += patTriggerEventContent
    for matchLabel in process.patTriggerEvent.patTriggerMatches:
        process.out.outputCommands += [ 'keep patTriggerObjectsedmAssociation_patTriggerEvent_' + matchLabel + '_*' ]

def switchOnTriggerStandAlone( process ):
    ## add trigger modules to path
    process.load("PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cff")
    process.patDefaultSequence += process.patTriggerSequence

    ## configure pat trigger
    process.patTrigger.onlyStandAlone = True
    process.patTriggerSequence.remove( process.patTriggerEvent )
    process.out.outputCommands += patTriggerStandAloneEventContent

def switchOnTriggerAll( process ):
    switchOnTrigger( process )
    process.out.outputCommands += patTriggerStandAloneEventContent

def switchOnTriggerMatchEmbedding( process ):
    process.patTriggerSequence += process.patTriggerMatchEmbedder
    process.out.outputCommands += patEventContentTriggerMatch
