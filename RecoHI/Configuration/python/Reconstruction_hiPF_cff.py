import FWCore.ParameterSet.Config as cms

# include  particle flow local reconstruction
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *

# run a trimmed down PF sequence with heavy-ion vertex, no conversions, nucl int, etc.

from RecoParticleFlow.PFProducer.particleFlowEGamma_cff import *
particleFlowEGamma.vertexCollection = "hiSelectedVertex"
gedGsfElectronCores.ctfTracks = "hiGeneralTracks"
gedGsfElectronsTmp.ctfTracksTag = "hiGeneralTracks"
gedGsfElectronsTmp.vtxTag = "hiSelectedVertex"
gedGsfElectronsTmp.preselection.minSCEtBarrel = 15.0
gedGsfElectronsTmp.preselection.minSCEtEndcaps = 15.0
gedGsfElectronsTmp.fillConvVtxFitProb = False

gedPhotonsTmp.primaryVertexProducer = "hiSelectedVertex"
gedPhotonsTmp.isolationSumsCalculatorSet.trackProducer = "hiGeneralTracks"
gedPhotons.primaryVertexProducer = "hiSelectedVertex"
gedPhotons.isolationSumsCalculatorSet.trackProducer = "hiGeneralTracks"
photonIDValueMaps.vertices = "hiSelectedVertex"
from RecoHI.HiEgammaAlgos.photonIsolationHIProducer_cfi import photonIsolationHIProducer

photonIsolationHIProducerGED = photonIsolationHIProducer.clone(photonProducer = "gedPhotonsTmp")

#These are set for consistency w/ HiElectronSequence, but these cuts need to be studied
gedGsfElectronsTmp.preselection.maxHOverEBarrelCone = 0.25
gedGsfElectronsTmp.preselection.maxHOverEEndcapsCone = 0.25
gedGsfElectronsTmp.preselection.maxHOverEBarrelBc = 0.0
gedGsfElectronsTmp.preselection.maxHOverEEndcapsBc = 0.0
gedGsfElectronsTmp.preselection.maxEOverPBarrel = 2.
gedGsfElectronsTmp.preselection.maxEOverPEndcaps = 2.

ootPhotonsTmp.primaryVertexProducer = "hiSelectedVertex"
ootPhotonsTmp.isolationSumsCalculatorSet.trackProducer = "hiGeneralTracks"
ootPhotons.primaryVertexProducer = "hiSelectedVertex"
ootPhotons.isolationSumsCalculatorSet.trackProducer = "hiGeneralTracks"

from RecoParticleFlow.Configuration.RecoParticleFlow_cff import *

mvaElectrons.vertexTag = "hiSelectedVertex"

particleFlowBlock.elementImporters = cms.VPSet(
    cms.PSet( importerName = cms.string("GSFTrackImporter"),
              source = cms.InputTag("pfTrackElec"),
              gsfsAreSecondary = cms.bool(False),
              superClustersArePF = cms.bool(True) ),
    cms.PSet( importerName = cms.string("SuperClusterImporter"),
                  source_eb = cms.InputTag("particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel"),
                  source_ee = cms.InputTag("particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower"),
                  source_towers = cms.InputTag("towerMaker"),
                  maximumHoverE = cms.double(0.5),
                  minSuperClusterPt = cms.double(10.0),
                  minPTforBypass = cms.double(100.0),
                  superClustersArePF = cms.bool(True) ),
    # all secondary track importers
    cms.PSet( importerName = cms.string("GeneralTracksImporter"),
              source = cms.InputTag("pfTrack"),
              muonSrc = cms.InputTag("hiMuons1stStep"),
              trackQuality = cms.string("highPurity"),
              cleanBadConvertedBrems = cms.bool(False),
              useIterativeTracking = cms.bool(False),
              DPtOverPtCuts_byTrackAlgo = cms.vdouble(-1.0,-1.0,-1.0,
                                                       1.0,1.0),
              NHitCuts_byTrackAlgo = cms.vuint32(3,3,3,3,3),
              muonMaxDPtOPt = cms.double(1),
              vetoEndcap = cms.bool(False)
              ),
    # to properly set SC based links you need to run ECAL importer
    # after you've imported all SCs to the block
    cms.PSet( importerName = cms.string("ECALClusterImporter"),
              source = cms.InputTag("particleFlowClusterECAL"),
              BCtoPFCMap = cms.InputTag('particleFlowSuperClusterECAL:PFClusterAssociationEBEE') ),
    cms.PSet( importerName = cms.string("GenericClusterImporter"),
              source = cms.InputTag("particleFlowClusterHCAL") ),
    cms.PSet( importerName = cms.string("GenericClusterImporter"),
              source = cms.InputTag("particleFlowClusterHO") ),
    cms.PSet( importerName = cms.string("GenericClusterImporter"),
              source = cms.InputTag("particleFlowClusterHF") ),
    cms.PSet( importerName = cms.string("GenericClusterImporter"),
              source = cms.InputTag("particleFlowClusterPS") )
    )

particleFlowTmp.postMuonCleaning = False
particleFlowTmp.vertexCollection = "hiSelectedVertex"
particleFlowTmp.muons = "hiMuons1stStep"
particleFlowTmp.usePFConversions = False

pfNoPileUpIso.enable = False
pfPileUpIso.Enable = False
pfNoPileUp.enable = False
pfPileUp.Enable = False
particleFlow.Muons = "muons:hiMuons1stStep2muonsMap"


# local reco must run before electrons (RecoHI/HiEgammaAlgos), due to PF integration
hiParticleFlowLocalRecoTask = cms.Task(particleFlowClusterTask)
hiParticleFlowLocalReco = cms.Sequence(hiParticleFlowLocalRecoTask)

particleFlowTmpTask = cms.Task(particleFlowTmp)
particleFlowTmpSeq = cms.Sequence(particleFlowTmpTask)

#PF Reco runs after electrons
hiParticleFlowRecoTask = cms.Task( pfGsfElectronMVASelectionTask
                                   , particleFlowBlock
                                   , particleFlowEGammaFullTask
                                   , photonIsolationHIProducerGED
                                   , particleFlowTmpTask
                                   , fixedGridRhoFastjetAllTmp
                                   , particleFlowTmpPtrs
                                   , particleFlowEGammaFinalTask
                                   , pfParticleSelectionTask
                                   )
hiParticleFlowReco = cms.Sequence(hiParticleFlowRecoTask)

particleFlowLinksTask = cms.Task( particleFlow,particleFlowPtrs,chargedHadronPFTrackIsolation,particleBasedIsolationTask)
particleFlowLinks = cms.Sequence(particleFlowLinksTask)
