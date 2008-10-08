import FWCore.ParameterSet.Config as cms

def switchLayer1Off(process):
    """ Disables trigger matching in PAT Layer 1 """
    process.allLayer1Electrons.addTrigMatch = False
    process.allLayer1Muons.addTrigMatch     = False
    process.allLayer1Jets.addTrigMatch      = False
    process.allLayer1METs.addTrigMatch      = False
    process.allLayer1Photons.addTrigMatch   = False
    process.allLayer1Taus.addTrigMatch      = False

