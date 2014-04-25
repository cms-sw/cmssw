import FWCore.ParameterSet.Config as cms

# include  particle flow local reconstruction
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *
particleFlowClusterPS.thresh_Pt_Seed_Endcap = cms.double(99999.)

from RecoParticleFlow.PFTracking.pfTrack_cfi import *
pfTrack.UseQuality = cms.bool(True)
pfTrack.TrackQuality = cms.string('highPurity')
pfTrack.TkColList = cms.VInputTag("hiSelectedTracks")
pfTrack.PrimaryVertexLabel = cms.InputTag("hiSelectedVertex")
pfTrack.MuColl = cms.InputTag("muons")

# run a trimmed down PF sequence with heavy-ion vertex, no conversions, nucl int, etc.
from RecoParticleFlow.Configuration.RecoParticleFlow_cff import *
particleFlowBlock.elementImporters = cms.VPSet(
    cms.PSet( importerName = cms.string("GSFTrackImporter"),
              source = cms.InputTag("pfTrackElec"),
              gsfsAreSecondary = cms.bool(False),
              superClustersArePF = cms.bool(True) ),        
    cms.PSet( importerName = cms.string("EGPhotonImporter"),
              source = cms.InputTag("mustachePhotons"),
              superClustersArePF = cms.bool(True),
              SelectionChoice = cms.string("CombinedDetectorIso"),
              SelectionDefinition = cms.PSet( minEt = cms.double(-99),
                                              # for SeperateDetectorIso
                                              trackIsoConstTerm = cms.double(2.0),
                                              trackIsoSlopeTerm = cms.double(0.001),
                                              ecalIsoConstTerm = cms.double(4.2),
                                              ecalIsoSlopeTerm = cms.double(0.003),
                                              hcalIsoConstTerm = cms.double(2.2),
                                              hcalIsoSlopeTerm = cms.double(0.001),
                                              HoverE = cms.double(0.05),
                                              #for CombinedDetectorIso
                                              LooseHoverE = cms.double(99999.0),
                                              combIsoConstTerm = cms.double(99999.0)
                                              ) 
              ),        
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
              source = cms.InputTag("particleFlowClusterHFEM") ),
    cms.PSet( importerName = cms.string("GenericClusterImporter"),
              source = cms.InputTag("particleFlowClusterHFHAD") ),
    cms.PSet( importerName = cms.string("GenericClusterImporter"),
              source = cms.InputTag("particleFlowClusterPS") )
    )

particleFlowTmp.postMuonCleaning = cms.bool(False)
particleFlowTmp.vertexCollection = cms.InputTag("hiSelectedVertex")
particleFlowTmp.usePFElectrons = cms.bool(True)
particleFlowTmp.muons = cms.InputTag("muons")
particleFlowTmp.usePFConversions = cms.bool(False)

from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *
pfTrackElec.applyGsfTrackCleaning = cms.bool(True)
pfTrackElec.PrimaryVertexLabel = cms.InputTag("hiSelectedVertex")

mvaElectrons.vertexTag = cms.InputTag("hiSelectedVertex")

# local reco must run before electrons (RecoHI/HiEgammaAlgos), due to PF integration
HiParticleFlowLocalReco = cms.Sequence(particleFlowCluster
                                       * pfTrack
                                       * pfTrackElec
                                       )

#PF Reco runs after electrons
HiParticleFlowReco = cms.Sequence(pfGsfElectronMVASelectionSequence
                                  * particleFlowBlock
                                  * particleFlowTmp
                                  )
