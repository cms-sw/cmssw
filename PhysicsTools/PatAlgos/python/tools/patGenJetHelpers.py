"""Plain helper functions of the pat::GenJet python configuration (no module is defined here, nothing is added to a process)."""
import FWCore.ParameterSet.Config as cms

STRIP = 'ForPatGenJets'  # postfix of every helper module label; the producers strip it

def userDataPSet(floats=(), ints=(), strip=(STRIP,)):
    """userData block of PATGenJetProducer with every PSet that PATUserDataHelper::fillDescription requires."""
    return cms.PSet(
        userFloats=cms.PSet(src=cms.VInputTag(*floats), labelPostfixesToStrip=cms.vstring(*strip)),
        userInts=cms.PSet(src=cms.VInputTag(*ints), labelPostfixesToStrip=cms.vstring(*strip)),
        userClasses=cms.PSet(src=cms.VInputTag()),
        userCands=cms.PSet(src=cms.VInputTag()),
        userFunctions=cms.vstring(),
        userFunctionLabels=cms.vstring(),
    )
