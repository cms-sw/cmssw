import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.jets_cff import *
from PhysicsTools.NanoAOD.muons_cff import *
from PhysicsTools.NanoAOD.taus_cff import *
from PhysicsTools.NanoAOD.electrons_cff import *
from PhysicsTools.NanoAOD.photons_cff import *
from PhysicsTools.NanoAOD.globals_cff import *
from PhysicsTools.NanoAOD.ttbarCategorization_cff import *
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
    preferredPDFs = cms.VPSet( # see https://lhapdf.hepforge.org/pdfsets.html
        cms.PSet( name = cms.string("PDF4LHC15_nnlo_30_pdfas"), lhaid = cms.uint32(91400) ),
        cms.PSet( name = cms.string("NNPDF31_nnlo_hessian_pdfas"), lhaid = cms.uint32(306000) ),
        cms.PSet( name = cms.string("NNPDF30_nlo_as_0118"), lhaid = cms.uint32(260000) ), # for some 92X samples. Note that the nominal weight, 260000, is not included in the LHE ...
        cms.PSet( name = cms.string("NNPDF30_lo_as_0130"), lhaid = cms.uint32(262000) ), # some MLM 80X samples have only this (e.g. /store/mc/RunIISummer16MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v2/120000/02A210D6-F5C3-E611-B570-008CFA197BD4.root )
    ),
    namedWeightIDs = cms.vstring(),
    namedWeightLabels = cms.vstring(),
    lheWeightPrecision = cms.int32(14),
    maxPdfWeights = cms.uint32(150), 
    debug = cms.untracked.bool(False),
)
lheInfoTable = cms.EDProducer("LHETablesProducer",
    lheInfo = cms.InputTag("externalLHEProducer"),
    precision = cms.int32(14),
    storeLHEParticles = cms.bool(True) 
)

l1bits=cms.EDProducer("L1TriggerResultsConverter", src=cms.InputTag("gtStage2Digis"), legacyL1=cms.bool(False))

nanoSequence = cms.Sequence(
        nanoMetadata + jetSequence + muonSequence + tauSequence + electronSequence+photonSequence+vertexSequence+metSequence+
        isoTrackSequence + # must be after all the leptons 
        linkedObjects  +
        jetTables + muonTables + tauTables + electronTables + photonTables +  globalTables +vertexTables+ metTables+simpleCleanerTable + triggerObjectTables + isoTrackTables +
	l1bits)

nanoSequenceMC = cms.Sequence(genParticleSequence + particleLevelSequence + nanoSequence + jetMC + muonMC + electronMC + photonMC + tauMC + metMC + ttbarCatMCProducers +  globalTablesMC + genWeightsTable + genParticleTables + particleLevelTables + lheInfoTable  + ttbarCategoryTable )


def nanoAOD_customizeCommon(process):
    return process

def nanoAOD_customizeData(process):
    process = nanoAOD_customizeCommon(process)
    if hasattr(process,'calibratedPatElectrons80X'):
        process.calibratedPatElectrons80X.isMC = cms.bool(False)
        process.calibratedPatPhotons80X.isMC = cms.bool(False)
    return process

def nanoAOD_customizeMC(process):
    process = nanoAOD_customizeCommon(process)
    if hasattr(process,'calibratedPatElectrons80X'):
        process.calibratedPatElectrons80X.isMC = cms.bool(True)
        process.calibratedPatPhotons80X.isMC = cms.bool(True)
    return process

### Era dependent customization
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
_80x_sequence = nanoSequence.copy()
#remove stuff 
_80x_sequence.remove(isoTrackTable)
_80x_sequence.remove(isoTrackSequence)

run2_miniAOD_80XLegacy.toReplaceWith( nanoSequence, _80x_sequence)

	

