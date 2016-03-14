import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.JetAnalysis.makePartons_cff import myPartons
myPartons.src = 'hiSignalGenParticles'

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
    ak1HiSignalGenJets + 
    ak2HiSignalGenJets +
    ak3HiSignalGenJets +
    ak4HiSignalGenJets +
    ak5HiSignalGenJets +
    ak6HiSignalGenJets )
