import FWCore.ParameterSet.Config as cms

# include  particle flow local reconstruction
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *

# run a trimmed down PF sequence with heavy-ion vertex, no conversions, nucl int, etc.

from RecoParticleFlow.PFProducer.particleFlowEGamma_cff import *
particleFlowEGamma.vertexCollection = cms.InputTag("hiSelectedVertex")
gedGsfElectronCores.ctfTracks = cms.InputTag("hiGeneralTracks")
gedGsfElectronsTmp.ctfTracksTag = cms.InputTag("hiGeneralTracks")
gedGsfElectronsTmp.vtxTag = cms.InputTag("hiSelectedVertex")
gedPhotonsTmp.primaryVertexProducer = cms.InputTag("hiSelectedVertex")
gedPhotonsTmp.regressionConfig.vertexCollection = cms.InputTag("hiSelectedVertex")
gedPhotonsTmp.isolationSumsCalculatorSet.trackProducer = cms.InputTag("hiGeneralTracks")


#These are set for consistency w/ HiElectronSequence, but these cuts need to be studied
gedGsfElectronsTmp.maxHOverEBarrel = cms.double(0.25)
gedGsfElectronsTmp.maxHOverEEndcaps = cms.double(0.25)


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
              muonSrc = cms.InputTag("muons"),
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
particleFlowTmp.usePFElectrons = cms.bool(True)
particleFlowTmp.muons = cms.InputTag("muons")
particleFlowTmp.usePFConversions = cms.bool(False)


from RecoHI.HiJetAlgos.HiRecoPFJets_cff import *

# local reco must run before electrons (RecoHI/HiEgammaAlgos), due to PF integration
hiParticleFlowLocalReco = cms.Sequence(particleFlowCluster)


#PF Reco runs after electrons
hiParticleFlowReco = cms.Sequence( pfGsfElectronMVASelectionSequence
                                   * particleFlowBlock
                                   * particleFlowEGammaFull
                                   * particleFlowTmp
                                   * hiRecoPFJets
                                   )
