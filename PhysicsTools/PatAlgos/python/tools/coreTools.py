import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import *

def switchMCAndTriggerMatch(process,fromCollection,toCollection):
    massSearchReplaceParam(process.patTrigMatch, 'src', fromCollection, toCollection)
    massSearchReplaceParam(process.patMCTruth,   'src', fromCollection, toCollection)


def removeSpecificPATObject(process,name):
    "Name should be something like 'Photons'"
    process.allLayer1Objects.remove( getattr(process, 'allLayer1'+name) )
    process.selectedLayer1Objects.remove( getattr(process, 'selectedLayer1'+name) )
    process.cleanLayer1Objects.remove( getattr(process, 'cleanLayer1'+name) )

    # counting
    process.countLayer1Objects.remove( getattr(process, 'countLayer1'+name) )
    # in the case of leptons, the lepton counter must be modified as well
    if name == 'Electrons':
        print 'removing electrons'
        process.countLayer1Leptons.countElectrons = False
    elif name == 'Muons':
        print 'removing muons - to be tested!'
        process.countLayer1Leptons.countMuons = False
    elif name == 'Taus':
        print 'removing taus - to be tested!'
        process.countLayer1Leptons.countTaus = False
    # remove from summary

    process.allLayer1Summary.candidates.remove( cms.InputTag('allLayer1'+name) )
    process.selectedLayer1Summary.candidates.remove( cms.InputTag('selectedLayer1'+name) )
    process.cleanLayer1Summary.candidates.remove( cms.InputTag('cleanLayer1'+name) )

def removeCleaning(process):
    for m in listModules(process.countLayer1Objects):
        if hasattr(m, 'src'): m.src = m.src.value().replace('cleanLayer1','selectedLayer1')
    countLept = process.countLayer1Leptons
    countLept.electronSource = countLept.electronSource.value().replace('cleanLayer1','selectedLayer1')
    countLept.muonSource     = countLept.muonSource.value().replace('cleanLayer1','selectedLayer1')
    countLept.tauSource      = countLept.tauSource.value().replace('cleanLayer1','selectedLayer1')
    process.patDefaultSequence.remove(process.cleanLayer1Objects)
