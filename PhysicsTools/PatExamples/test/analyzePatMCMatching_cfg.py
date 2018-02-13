import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

#------------------------------------------------------------------------------------------------------
# To configure the Matching, we have to configure the PAT-Workflow starting from the patDefaultSequence:
#------------------------------------------------------------------------------------------------------

from PhysicsTools.PatAlgos.patTemplate_cfg import *

## increase the number of events a bit
process.maxEvents.input = 1000

## add inFlightMuons
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.inFlightMuons = cms.EDProducer("PATGenCandsFromSimTracksProducer",
        src           = cms.InputTag("g4SimHits"),   ## use "famosSimHits" for FAMOS
        setStatus     = cms.int32(-1),
        particleTypes = cms.vstring("mu+"),          ## picks also mu-, of course
        filter        = cms.vstring("pt > 0.5"),     ## just for testing
        makeMotherLink = cms.bool(True),
        writeAncestors = cms.bool(True),             ## save also the intermediate GEANT ancestors of the muons
        genParticles   = cms.InputTag("genParticles"),
)
## prepare several clones of match associations for status 1, 3 and in flight muons (status -1)
process.muMatch3 = process.muonMatch.clone(mcStatus = [3]) # hard scattering
process.muMatch1 = process.muonMatch.clone(mcStatus = [1]) # stable


## add the new matches to the default sequence
process.patDefaultSequence.replace(process.muonMatch,
                                   process.muMatch1 +
                                   process.muMatch3
)

process.patMuons.genParticleMatch = cms.VInputTag(
    cms.InputTag("muMatch3"),
    cms.InputTag("muMatch1")
)


#-----------------------------------------
# As usual add those two usefull things:
#----------------------------------------

process.TFileService = cms.Service("TFileService",
  fileName = cms.string('analyzePatMCMatching.root')
)

process.MessageLogger = cms.Service("MessageLogger")


#----------------------------------------------------------------------
# Finally let's analyze the matching and run all that in correct order:
#----------------------------------------------------------------------

process.analyzePatMCMatching = cms.EDAnalyzer("PatMCMatching",
  muonSrc     = cms.untracked.InputTag("cleanPatMuons")                                             
)


process.outpath.remove(process.out)

process.p = cms.Path(process.patDefaultSequence + process.analyzePatMCMatching)

del(process.out)
del(process.outpath)

