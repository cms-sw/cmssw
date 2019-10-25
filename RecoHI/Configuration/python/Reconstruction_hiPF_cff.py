import FWCore.ParameterSet.Config as cms

# include  particle flow local reconstruction
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *

# run a trimmed down PF sequence with heavy-ion vertex, no conversions, nucl int, etc.

from RecoParticleFlow.PFProducer.particleFlowEGamma_cff import *
particleFlowEGamma.vertexCollection = cms.InputTag("hiSelectedVertex")
gedGsfElectronCores.ctfTracks = cms.InputTag("hiGeneralTracks")
gedGsfElectronsTmp.ctfTracksTag = cms.InputTag("hiGeneralTracks")
gedGsfElectronsTmp.vtxTag = cms.InputTag("hiSelectedVertex")
gedGsfElectronsTmp.preselection.minSCEtBarrel = cms.double(15.0)
gedGsfElectronsTmp.preselection.minSCEtEndcaps = cms.double(15.0)
gedGsfElectronsTmp.fillConvVtxFitProb = cms.bool(False)

gedPhotonsTmp.primaryVertexProducer = cms.InputTag("hiSelectedVertex")
gedPhotonsTmp.isolationSumsCalculatorSet.trackProducer = cms.InputTag("hiGeneralTracks")
gedPhotons.primaryVertexProducer = cms.InputTag("hiSelectedVertex")
gedPhotons.isolationSumsCalculatorSet.trackProducer = cms.InputTag("hiGeneralTracks")
photonIDValueMaps.vertices = cms.InputTag("hiSelectedVertex")
from RecoHI.HiEgammaAlgos.photonIsolationHIProducer_cfi import photonIsolationHIProducer
photonIsolationHIProducerGED = photonIsolationHIProducer.clone(photonProducer=cms.InputTag("gedPhotonsTmp"))

#These are set for consistency w/ HiElectronSequence, but these cuts need to be studied
gedGsfElectronsTmp.preselection.maxHOverEBarrelCone = cms.double(0.25)
gedGsfElectronsTmp.preselection.maxHOverEEndcapsCone = cms.double(0.25)
gedGsfElectronsTmp.preselection.maxHOverEBarrelTower = cms.double(0.0)
gedGsfElectronsTmp.preselection.maxHOverEEndcapsTower = cms.double(0.0)
gedGsfElectronsTmp.preselection.maxEOverPBarrel = cms.double(2.)
gedGsfElectronsTmp.preselection.maxEOverPEndcaps = cms.double(2.)

ootPhotonsTmp.primaryVertexProducer = cms.InputTag("hiSelectedVertex")
ootPhotonsTmp.isolationSumsCalculatorSet.trackProducer = cms.InputTag("hiGeneralTracks")
ootPhotons.primaryVertexProducer = cms.InputTag("hiSelectedVertex")
ootPhotons.isolationSumsCalculatorSet.trackProducer = cms.InputTag("hiGeneralTracks")

from RecoParticleFlow.Configuration.RecoParticleFlow_cff import *

particleFlowClusterECAL.energyCorrector.verticesLabel = cms.InputTag('hiPixelAdaptiveVertex')

mvaElectrons.vertexTag = cms.InputTag("hiSelectedVertex")

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
              useIterativeTracking = cms.bool(False),
              DPtOverPtCuts_byTrackAlgo = cms.vdouble(-1.0,-1.0,-1.0,
                                                       1.0,1.0),
              NHitCuts_byTrackAlgo = cms.vuint32(3,3,3,3,3)
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

particleFlowTmp.postMuonCleaning = cms.bool(False)
particleFlowTmp.vertexCollection = cms.InputTag("hiSelectedVertex")
particleFlowTmp.muons = cms.InputTag("hiMuons1stStep")
particleFlowTmp.usePFConversions = cms.bool(False)

pfNoPileUpIso.enable = False
pfPileUpIso.Enable = False
pfNoPileUp.enable = False
pfPileUp.Enable = False
particleFlow.Muons = cms.InputTag("muons","hiMuons1stStep2muonsMap")


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
