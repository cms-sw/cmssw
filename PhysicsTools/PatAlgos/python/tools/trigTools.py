import FWCore.ParameterSet.Config as cms

def switchOffTriggerMatchingOld( process ):
    """ Disables old style trigger matching in PAT  """
    process.patDefaultSequence.remove( process.patTrigMatch )
    process.allLayer1Electrons.addTrigMatch = False
    process.allLayer1Muons.addTrigMatch     = False
    process.allLayer1Jets.addTrigMatch      = False
    process.allLayer1Photons.addTrigMatch   = False
    process.allLayer1Taus.addTrigMatch      = False
    process.layer1METs.addTrigMatch         = False
    
# for (temporary) backwards compatibility
def switchTriggerOff( process ):
    switchOffTriggerMatchingOld( process )

from PhysicsTools.PatAlgos.patEventContent_cff import patTriggerEventContent, patTriggerStandAloneEventContent
    
def switchOnTrigger( process ):
    """ Enables trigger information in PAT  """
    process.patTrigger.onlyStandAlone = False
    # add trigger modules to path
    process.p *= process.patTriggerSequence
    # add trigger specific event content to PAT event content
    process.out.outputCommands += patTriggerEventContent
    for matchLabel in process.patTriggerEvent.patTriggerMatches:
        process.out.outputCommands += [ 'keep patTriggerObjectsedmAssociation_patTriggerEvent' + matchLabel + '_*' ]

def switchOnTriggerStandAlone( process ):
    process.patTrigger.onlyStandAlone = True
    process.patTriggerSequence.remove( process.patTriggerEvent )
    process.p *= process.patTriggerSequence
    process.out.outputCommands += patTriggerStandAloneEventContent

def switchOnTriggerAll( process ):
    switchOnTrigger( process )
    process.out.outputCommands += patTriggerStandAloneEventContent
