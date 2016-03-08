import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.JetAnalysis.makePartons_cff import myPartons
from RecoHI.HiJetAlgos.HiGenCleaner_cff import  hiPartons

myPartons.src = 'hiSignalGenParticles'
selectedPartons = hiPartons.clone(src = 'myPartons')
# matcher doesn't like to use the parton collection directly for some reason.  Hand it the cleaned collection w/ cleaning turned off instead.
selectedPartons.deltaR  = -1.
selectedPartons.ptCut  = -1.

from RecoHI.HiJetAlgos.HiSignalParticleProducer_cfi import hiSignalGenParticles
from RecoHI.HiJetAlgos.HiSignalGenJetProducer_cfi import hiSignalGenJets

ak1HiSignalGenJets = hiSignalGenJets.clone(src = "ak1HiGenJets") 
ak2HiSignalGenJets = hiSignalGenJets.clone(src = "ak2HiGenJets") 
ak3HiSignalGenJets = hiSignalGenJets.clone(src = "ak3HiGenJets") 
ak4HiSignalGenJets = hiSignalGenJets.clone(src = "ak4HiGenJets") 
ak5HiSignalGenJets = hiSignalGenJets.clone(src = "ak5HiGenJets") 
ak6HiSignalGenJets = hiSignalGenJets.clone(src = "ak6HiGenJets") 

hiSignalGenFilters = cms.Sequence(
    hiSignalGenParticles + 
    myPartons + 
    selectedPartons +
    ak1HiSignalGenJets + 
    ak2HiSignalGenJets +
    ak3HiSignalGenJets +
    ak4HiSignalGenJets +
    ak5HiSignalGenJets +
    ak6HiSignalGenJets )
