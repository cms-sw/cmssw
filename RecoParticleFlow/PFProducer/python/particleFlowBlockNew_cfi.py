import FWCore.ParameterSet.Config as cms

particleFlowBlockNew = cms.EDProducer(
    "PFBlockProducerNew",
    # verbosity 
    verbose = cms.untracked.bool(False),
    # Debug flag
    debug = cms.untracked.bool(False),

    # Link tracks and HCAL clusters to HO clusters
    useHO = cms.bool(True),

    #define what we are importing into particle flow
    #from the various subdetectors
    # importers are executed in the order they are defined here!!!
    #order matters for some modules (it is pointed out where this is important)
    # you can find a list of all available importers in:
    #  plugins/importers
    elementImporters = cms.VPSet(
        cms.PSet( importerName = cms.string("GSFTrackImporter"),
                  source = cms.InputTag("pfTrackElec"),
                  gsfsAreSecondary = cms.bool(False) ),
        cms.PSet( importerName = cms.string("ConvBremTrackImporter"),
                  source = cms.InputTag("pfTrackElec") ),
        cms.PSet( importerName = cms.string("EGPhotonImporter"),
                  source = cms.InputTag("mustachePhotons"),
                  SelectionChoice = cms.string("CombinedDetectorIso"),
                  SelectionDefinition = cms.PSet( 
                             minEt = cms.double(-99),
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
                             ) ),        
        cms.PSet( importerName = cms.string("ConversionTrackImporter"),
                  source = cms.InputTag("pfConversions") ),
        # V0's not actually used in particle flow block building so far
        #cms.PSet( importerName = cms.string("V0TrackImporter"),
        #          source = cms.InputTag("pfV0") ),
        #NuclearInteraction's also come in Loose and VeryLoose varieties
        cms.PSet( importerName = cms.string("NuclearInteractionTrackImporter"),
                  source = cms.InputTag("pfDisplacedTrackerVertex") ),
        #for best timing GeneralTracksImporter should come after
        # all secondary track importers
        cms.PSet( importerName = cms.string("GeneralTracksImporter"),
                  source = cms.InputTag("pfTrack"),
                  muonSrc = cms.InputTag("muons1stStep"),
                  useIterativeTracking = cms.bool(True),
                  DPtOverPtCuts_byTrackAlgo = cms.vdouble(-1.0,-1.0,-1.0,
                                                           1.0,1.0),
                  NHitCuts_byTrackAlgo = cms.vuint32(3,3,3,3,3)
                  ),
        # secondary GSF tracks are also turned off
        #cms.PSet( importerName = cms.string("GSFTrackImporter"),
        #          source = cms.InputTag("pfTrackElec:Secondary"),
        #          gsfsAreSecondary = cms.bool(True) ),
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
                  source = cms.InputTag("particleFlowClusterPS") ),
        
        ),
    
    #linking definitions
    # you can find a list of all available linkers in:
    #  plugins/linkers
    linkDefinitions = cms.VPSet(
        cms.PSet( linkerName = cms.string("PreshowerAndECALLinker"),
                  linkType   = cms.string("PS1:ECAL"),
                  useKDTree  = cms.bool(True) ),
        cms.PSet( linkerName = cms.string("PreshowerAndECALLinker"),
                  linkType   = cms.string("PS2:ECAL"),
                  useKDTree  = cms.bool(True) ),
        cms.PSet( linkerName = cms.string("TrackAndECALLinker"),
                  linkType   = cms.string("TRACK:ECAL"),
                  useKDTree  = cms.bool(True) ),
        cms.PSet( linkerName = cms.string("TrackAndHCALLinker"),
                  linkType   = cms.string("TRACK:HCAL"),
                  useKDTree  = cms.bool(True) ),
        cms.PSet( linkerName = cms.string("TrackAndHOLinker"),
                  linkType   = cms.string("TRACK:HO"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("ECALAndHCALLinker"),
                  linkType   = cms.string("ECAL:HCAL"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("HCALAndHOLinker"),
                  linkType   = cms.string("HCAL:HO"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("HFEMAndHFHADLinker"),
                  linkType   = cms.string("HFEM:HFHAD"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("TrackAndTrackLinker"),
                  linkType   = cms.string("TRACK:TRACK"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("ECALAndECALLinker"),
                  linkType   = cms.string("ECAL:ECAL"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("GSFAndECALLinker"), 
                  linkType   = cms.string("GSF:ECAL"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("TrackAndGSFLinker"),
                  linkType   = cms.string("TRACK:GSF"),
                  useKDTree  = cms.bool(False),
                  useConvertedBrems = cms.bool(True) ),
        cms.PSet( linkerName = cms.string("GSFAndBREMLinker"),# here
                  linkType   = cms.string("GSF:BREM"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("GSFAndGSFLinker"),
                  linkType   = cms.string("GSF:GSF"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("ECALAndBREMLinker"),
                  linkType   = cms.string("ECAL:BREM"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("GSFAndHCALLinker"),
                  linkType   = cms.string("GSF:HCAL"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("HCALAndBREMLinker"),
                  linkType   = cms.string("HCAL:BREM"),
                  useKDTree  = cms.bool(False) ),
        cms.PSet( linkerName = cms.string("SCAndECALLinker"),
                  linkType   = cms.string("SC:ECAL"),
                  useKDTree  = cms.bool(False),
                  SuperClusterMatchByRef = cms.bool(True) )
        ),
                                      
    # input clusters
    PFClustersECAL = cms.InputTag("particleFlowClusterECAL"),
    PFClustersHCAL = cms.InputTag("particleFlowClusterHCAL"),
    PFClustersHO = cms.InputTag("particleFlowClusterHO"),	
    # For upgrade studies:
#    PFClustersHCAL = cms.InputTag("particleFlowHCALSuperClusterDualTime"),
    PFClustersHFEM = cms.InputTag("particleFlowClusterHFEM"),
    PFClustersHFHAD = cms.InputTag("particleFlowClusterHFHAD"),
    PFClustersPS = cms.InputTag("particleFlowClusterPS"),
    EGPhotons = cms.InputTag("mustachePhotons"),  
    #disable dierct import of SuperCluster collections for now until effect on blocks can be
    #evaluated
    useSuperClusters = cms.bool(False),
    #current egamma superclusters
    SCBarrel = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel'),
    SCEndcap = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower'),
    #pfbox superclusters, will switch to this in the near future
    #SCBarrel = cms.InputTag("particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel"),                                   
    #SCEndcap = cms.InputTag("particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower"), 
    
    #since we are using ParticleFlow SuperClusters everywhere, match to PFClusters by reference using
    #the required ValueMap
    #n.b This requires that PF SuperCluseters are used for both the directly imported SuperClusters or photons
    #AND for the imported electron seeds
    SuperClusterMatchByRef = cms.bool(True),
    PFClusterAssociationEBEE = cms.InputTag('particleFlowSuperClusterECAL:PFClusterAssociationEBEE'),
    
    # input tracks
    GsfRecTracks = cms.InputTag("pfTrackElec"),
    ConvBremGsfRecTracks = cms.InputTag("pfTrackElec","Secondary"),
    useConvBremGsfTracks = cms.bool(False),                                     
    RecTracks = cms.InputTag("pfTrack"),
    useConvBremPFRecTracks = cms.bool(True),

    # input nuclear interactions 
    PFNuclear = cms.InputTag("pfDisplacedTrackerVertex"),
    useNuclear = cms.bool(True),

    # This parameters defines the level of purity of
    # nuclear interactions choosen.
    # Level 1 is only high Purity sample labeled as isNucl
    # Level 2 isNucl + isNucl_Loose (2 secondary tracks vertices)
    # Level 3 isNucl + isNucl_Loose + isNucl_Kink
    #         (low purity sample made of 1 primary and 1 secondary track)
    # By default the level 1 is teh safest one.

    nuclearInteractionsPurity = cms.uint32(1),                          

    # input muons
    RecMuons = cms.InputTag("muons1stStep"),

    # input conversions
    PFConversions = cms.InputTag("pfConversions"),
    useConversions = cms.bool(True),

    # Glowinski & Gouzevitch                             
    useKDTreeTrackEcalLinker = cms.bool(True),

    # input V0
    PFV0 = cms.InputTag("pfV0"),
    useV0 = cms.bool(False),

    # Track Quality Cut: Tracks are kept if DPt/Pt < sigma * Cut
    # and if nHit >= cut
    pf_DPtoverPt_Cut = cms.vdouble(-1.0,-1.0,-1.0,1.0,1.0),
    pf_NHit_Cut = cms.vuint32(3,3,3,3,3),
                                   
    # Run particle flow at HLT (hence no RecMuons, no GSF tracks)
    usePFatHLT = cms.bool(False),

    # Turn of track quality cuts that require iterative tracking for heavy-ions
    useIterTracking = cms.bool(True),

    # Photon selection. Et cut; Track iso (cste;slope), Ecal iso (cste, slope), Hcal iso (cste, slope), H/E
    # just put infinite Et cut to disable the photon import
    useEGPhotons = cms.bool(True),                                   
    PhotonSelectionCuts = cms.vdouble(1,-99.,2.0, 0.001, 4.2, 0.003, 2.2, 0.001, 0.05, 99999., 99999.)
)


