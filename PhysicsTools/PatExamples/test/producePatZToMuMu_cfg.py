## Starting with a skeleton process which gets imported with the following line
from PhysicsTools.PatAlgos.patTemplate_cfg import *

## ---
## Use proper input
## ---
from PhysicsTools.PatExamples.samplesCERN_cff import *
process.source.fileNames = zjetsRECO


## ---
## Adjust inputs if necessary
## ---
#from PhysicsTools.PatAlgos.tools.cmsswVersionTools import run36xOn35xInput
#run36xOn35xInput(process)

## This might be needed when running on 383 rereco'ed data
#process.load("RecoJets.Configuration.GenJetParticles_cff")
#process.load("RecoJets.Configuration.RecoGenJets_cff")

#process.p0 = cms.Path(
#    process.genJetParticles *
#    process.ak5GenJets
#)

## ---
## Determine number of events to be processed
## ---
process.maxEvents.input = 100

## ---
## Adaptations to the event content
## ---
process.p = cms.Path(
    process.patDefaultSequence
)

## Switch embedding to false
process.patMuons.embedStandAloneMuon = False
process.patMuons.embedCombinedMuon = False
process.patMuons.embedTrack = False

## Keep tracks and track extras instead
process.out.outputCommands+= [
    "keep *_globalMuons_*_*",
    "keep *_generalTracks_*_*",
    "keep *_standAloneMuons_*_*"
    ]
