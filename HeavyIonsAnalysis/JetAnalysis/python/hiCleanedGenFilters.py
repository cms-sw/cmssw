import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.JetAnalysis.makePartons_cff import myPartons
from RecoHI.HiJetAlgos.HiGenCleaner_cff import heavyIonCleanedGenJets, hiPartons
selectedPartons = hiPartons.clone(src = 'myPartons')

ak1HiCleanedGenJets = heavyIonCleanedGenJets.clone(src = "ak1HiGenJets") 
ak2HiCleanedGenJets = heavyIonCleanedGenJets.clone(src = "ak2HiGenJets") 
ak3HiCleanedGenJets = heavyIonCleanedGenJets.clone(src = "ak3HiGenJets") 
ak4HiCleanedGenJets = heavyIonCleanedGenJets.clone(src = "ak4HiGenJets") 
ak5HiCleanedGenJets = heavyIonCleanedGenJets.clone(src = "ak5HiGenJets") 
ak6HiCleanedGenJets = heavyIonCleanedGenJets.clone(src = "ak6HiGenJets") 

hiCleanedGenFilters = cms.Sequence(
    myPartons + 
    selectedPartons + 
    ak1HiCleanedGenJets + 
    ak2HiCleanedGenJets +
    ak3HiCleanedGenJets +
    ak4HiCleanedGenJets +
    ak5HiCleanedGenJets +
    ak6HiCleanedGenJets )
