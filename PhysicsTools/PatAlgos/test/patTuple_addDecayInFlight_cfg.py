## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

#process.Tracer = cms.Service("Tracer")

# load the PAT config
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
patAlgosToolsTask.add(process.patCandidatesTask)

process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")
patAlgosToolsTask.add(process.selectedPatCandidatesTask)

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
patAlgosToolsTask.add(process.inFlightMuons)

process.out.outputCommands.append('keep *_inFlightMuons_*_*')

## prepare several clones of match associations for status 1, 3 and in flight muons (status -1)
process.muMatch3 = process.muonMatch.clone(mcStatus = cms.vint32( 3))
patAlgosToolsTask.add(process.muMatch3)

process.muMatch1 = process.muonMatch.clone(mcStatus = cms.vint32( 1))
patAlgosToolsTask.add(process.muMatch1)

process.muMatchF = process.muonMatch.clone(mcStatus = cms.vint32(-1),matched = cms.InputTag("inFlightMuons"))
patAlgosToolsTask.add(process.muMatchF)

process.patMuons.genParticleMatch = cms.VInputTag(
    cms.InputTag("muMatch3"),
    cms.InputTag("muMatch1"),
    cms.InputTag("muMatchF"),
)

## dump event content
process.content = cms.EDAnalyzer("EventContentAnalyzer")

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
## switch to RECO input
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarGENSIMRECO
process.source.fileNames = filesRelValTTbarGENSIMRECO
#                                         ##
process.maxEvents.input = 10
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_addDecayInFlight.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
