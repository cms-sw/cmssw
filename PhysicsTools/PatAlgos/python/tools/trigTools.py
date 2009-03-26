import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.patEventContent_cff import patTriggerEventContent

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
    
def switchOnTrigger( process ):
    """ Enables trigger information in PAT  """
    # add trigger modules to path
    process.p *= process.patTriggerSequence
    # add trigger specific event content to PAT event content
    process.out.outputCommands += patTriggerEventContent
    for matchLabel in process.patTriggerEvent.patTriggerMatches:
        process.out.outputCommands += [ 'keep patTriggerObjectsedmAssociation_' + matchLabel + '_*_*' ]
