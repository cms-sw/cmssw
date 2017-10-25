import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.jets_cff import *
from PhysicsTools.NanoAOD.muons_cff import *
from PhysicsTools.NanoAOD.taus_cff import *
from PhysicsTools.NanoAOD.electrons_cff import *
from PhysicsTools.NanoAOD.photons_cff import *
from PhysicsTools.NanoAOD.globals_cff import *
from PhysicsTools.NanoAOD.genparticles_cff import *
from PhysicsTools.NanoAOD.vertices_cff import *
from PhysicsTools.NanoAOD.met_cff import *
from PhysicsTools.NanoAOD.triggerObjects_cff import *
from PhysicsTools.NanoAOD.isotracks_cff import *
from PhysicsTools.NanoAOD.NanoAODEDMEventContent_cff import *

nanoMetadata = cms.EDProducer("UniqueStringProducer",
    strings = cms.PSet(
        tag = cms.string("untagged"),
    )
)

linkedObjects = cms.EDProducer("PATObjectCrossLinker",
   jets=cms.InputTag("finalJets"),
   muons=cms.InputTag("finalMuons"),
   electrons=cms.InputTag("finalElectrons"),
   taus=cms.InputTag("finalTaus"),
   photons=cms.InputTag("finalPhotons"),
)

simpleCleanerTable = cms.EDProducer("NanoAODSimpleCrossCleaner",
   name=cms.string("cleanmask"),
   doc=cms.string("simple cleaning mask with priority to leptons"),
   jets=cms.InputTag("linkedObjects","jets"),
   muons=cms.InputTag("linkedObjects","muons"),
   electrons=cms.InputTag("linkedObjects","electrons"),
   taus=cms.InputTag("linkedObjects","taus"),
   photons=cms.InputTag("linkedObjects","photons"),
   jetSel=cms.string("pt>15"),
   muonSel=cms.string("isPFMuon && innerTrack.validFraction >= 0.49 && ( isGlobalMuon && globalTrack.normalizedChi2 < 3 && combinedQuality.chi2LocalPosition < 12 && combinedQuality.trkKink < 20 && segmentCompatibility >= 0.303 || segmentCompatibility >= 0.451 )"),
   electronSel=cms.string(""),
   tauSel=cms.string(""),
   photonSel=cms.string(""),
   jetName=cms.string("Jet"),muonName=cms.string("Muon"),electronName=cms.string("Electron"),
   tauName=cms.string("Tau"),photonName=cms.string("Photon")
)


genWeightsTable = cms.EDProducer("GenWeightsTableProducer",
    genEvent = cms.InputTag("generator"),
    lheInfo = cms.InputTag("externalLHEProducer"),
    preferredPDFs = cms.vuint32(91400,260001),
    namedWeightIDs = cms.vstring(),
    namedWeightLabels = cms.vstring(),
    lheWeightPrecision = cms.int32(14),
    maxPdfWeights = cms.uint32(50), # for NNPDF, keep only the first 50 replicas (save space)
    debug = cms.untracked.bool(False),
)
lheInfoTable = cms.EDProducer("LHETablesProducer",
    lheInfo = cms.InputTag("externalLHEProducer"),
)

l1bits=cms.EDProducer("L1TriggerResultsConverter", src=cms.InputTag("gtStage2Digis"), legacyL1=cms.bool(False))

nanoSequence = cms.Sequence(
        nanoMetadata + muonSequence + jetSequence + tauSequence + electronSequence+photonSequence+vertexSequence+metSequence+
        isoTrackSequence + # must be after all the leptons 
        linkedObjects  +
        jetTables + muonTables + tauTables + electronTables + photonTables +  globalTables +vertexTables+ metTables+simpleCleanerTable + triggerObjectTables + isoTrackTables +
	l1bits)

nanoSequenceMC = cms.Sequence(genParticleSequence + nanoSequence + jetMC + muonMC + electronMC + photonMC + tauMC + metMC + globalTablesMC + genWeightsTable + genParticleTables + lheInfoTable)


def nanoAOD_customizeCommon(process):
    return process

def nanoAOD_customizeData(process):
    process = nanoAOD_customizeCommon(process)
    process.calibratedPatElectrons.isMC = cms.bool(False)
    process.calibratedPatPhotons.isMC = cms.bool(False)
    return process

def nanoAOD_customizeMC(process):
    process = nanoAOD_customizeCommon(process)
    ## FIXME:  WILL NO LONGER NEED RANDOM SEEDS WHEN DETERMINISTIC SMEARING WILL BE IMPLEMENTED
    if not hasattr(process,'RandomNumberGeneratorService'):
        process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService")
    for X in 'calibratedPatElectrons','calibratedPatPhotons':
        if not hasattr(process.RandomNumberGeneratorService,X):
            setattr(process.RandomNumberGeneratorService, X, 
                cms.PSet(initialSeed = cms.untracked.uint32(81), engineName = cms.untracked.string('TRandom3')))
    process.calibratedPatElectrons.isMC = cms.bool(True)
    process.calibratedPatPhotons.isMC = cms.bool(True)
    return process

### Era dependent customization
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
#remove stuff 
_80x_sequence = nanoSequence.copy()
_80x_sequence.remove(isoTrackTable)
_80x_sequence.remove(isoTrackSequence)
run2_miniAOD_80XLegacy.toReplaceWith( nanoSequence, _80x_sequence)

	

