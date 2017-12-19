import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.jets_cff import *
from PhysicsTools.NanoAOD.muons_cff import *
from PhysicsTools.NanoAOD.taus_cff import *
from PhysicsTools.NanoAOD.electrons_cff import *
from PhysicsTools.NanoAOD.photons_cff import *
from PhysicsTools.NanoAOD.globals_cff import *
from PhysicsTools.NanoAOD.genparticles_cff import *
from PhysicsTools.NanoAOD.particlelevel_cff import *
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
    preferredPDFs = cms.vuint32(91400,260001,262001),
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

nanoSequenceMC = cms.Sequence(genParticleSequence + particleLevelSequence + nanoSequence + jetMC + muonMC + electronMC + photonMC + tauMC + metMC + globalTablesMC + genWeightsTable + genParticleTables + particleLevelTables + lheInfoTable)


def nanoAOD_customizeCommon(process):
    return process

def nanoAOD_customizeData(process):
    process = nanoAOD_customizeCommon(process)
    process.calibratedPatElectrons.isMC = cms.bool(False)
    process.calibratedPatPhotons.isMC = cms.bool(False)
    return process

def nanoAOD_customizeMC(process):
    process = nanoAOD_customizeCommon(process)
    process.calibratedPatElectrons.isMC = cms.bool(True)
    process.calibratedPatPhotons.isMC = cms.bool(True)
    return process

### Era dependent customization
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
from RecoJets.JetProducers.QGTagger_cfi import  QGTagger
qgtagger80x=QGTagger.clone(srcJets="slimmedJets",srcVertexCollection="offlineSlimmedPrimaryVertices")
_80x_sequence = nanoSequence.copy()
#remove stuff 
_80x_sequence.remove(isoTrackTable)
_80x_sequence.remove(isoTrackSequence)
#add qgl
_80x_sequence.insert(1,qgtagger80x)

_80x_sequenceMC = nanoSequenceMC.copy()
_80x_sequenceMC.remove(genSubJetAK8Table)
_80x_sequenceMC.remove(genJetFlavourTable)
_80x_sequenceMC.insert(-1,genJetFlavourAssociation)
_80x_sequenceMC.insert(-1,genJetFlavourTable)
run2_miniAOD_80XLegacy.toReplaceWith( nanoSequence, _80x_sequence)
run2_miniAOD_80XLegacy.toReplaceWith( nanoSequenceMC, _80x_sequenceMC)

	

from Configuration.Eras.Modifier_run2_nanoAOD_92X_cff import run2_nanoAOD_92X
#remove stuff

_92x_sequence = nanoSequence.copy()
_92x_sequenceMC = nanoSequenceMC.copy()
_92x_sequenceMC.remove(genSubJetAK8Table)
_92x_sequenceMC.remove(genJetFlavourTable)
_92x_sequenceMC.insert(-1,genJetFlavourAssociation)
_92x_sequenceMC.insert(-1,genJetFlavourTable)
run2_nanoAOD_92X.toReplaceWith( nanoSequence, _92x_sequence)
run2_nanoAOD_92X.toReplaceWith( nanoSequenceMC, _92x_sequenceMC)
