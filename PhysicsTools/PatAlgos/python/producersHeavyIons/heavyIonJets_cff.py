import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.GenJetParticles_cff import genParticlesForJets
from RecoHI.HiJetAlgos.HiGenCleaner_cff import hiPartons

allPartons = cms.EDProducer(
    "PartonSelector",
    src = cms.InputTag('genParticles'),
    withLeptons = cms.bool(False),
    )

cleanedPartons = hiPartons.clone(
    src = 'allPartons',
    )

cleanedGenJetsTask = cms.Task(
    genParticlesForJets,
    cleanedPartons,
)

from RecoHI.HiJetAlgos.HiRecoPFJets_cff import PFTowers, pfNoPileUpJMEHI, ak4PFJetsForFlow, hiPuRho, hiFJRhoFlowModulation
from RecoHI.HiTracking.highPurityTracks_cfi import highPurityTracks

recoPFJetsHIpostAODTask = cms.Task(
    PFTowers,
    pfNoPileUpJMEHI,
    ak4PFJetsForFlow,
    hiFJRhoFlowModulation,
    hiPuRho,
    highPurityTracks,
    )

recoJetsHIpostAODTask = cms.Task(
    recoPFJetsHIpostAODTask,
    allPartons,
    cleanedGenJetsTask,
    )
