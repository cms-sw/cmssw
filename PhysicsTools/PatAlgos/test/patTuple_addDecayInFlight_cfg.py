## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

## switch to RECO input
process.source.fileNames = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_2_0_pre6'
                        , relVal        = 'RelValProdTTbar'
                        , globalTag     = 'START52_V2'
                        , dataTier      = 'GEN-SIM-RECO'
                        , maxVersions   = 3
                        , numberOfFiles = 1
                        )
    )

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
process.muMatch3 = process.muonMatch.clone(mcStatus = cms.vint32( 3))
process.muMatch1 = process.muonMatch.clone(mcStatus = cms.vint32( 1))
process.muMatchF = process.muonMatch.clone(mcStatus = cms.vint32(-1),matched = cms.InputTag("inFlightMuons"))

## add the new matches to the default sequence
process.patDefaultSequence.replace(process.muonMatch,
                                   process.muMatch1 +
                                   process.muMatch3 +
                                   process.muMatchF
                                   )

## embed the new matches to the patMuon (they are then accessible via
## genMuon(int idx))
process.patMuons.genParticleMatch = cms.VInputTag(
    cms.InputTag("muMatch3"),
    cms.InputTag("muMatch1"),
    cms.InputTag("muMatchF"),
)

## dump event content
process.content = cms.EDAnalyzer("EventContentAnalyzer")

## add the in flight muons to the output
process.out.outputCommands.append('keep *_inFlightMuons_*_*')

## let it run
process.p = cms.Path(
    #process.content +
    process.inFlightMuons +
    process.patDefaultSequence
)
