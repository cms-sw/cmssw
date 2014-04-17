import FWCore.ParameterSet.Config as cms

particleFlowBlockNew = cms.EDProducer(
    "PFBlockProducerNew",
    # verbosity 
    verbose = cms.untracked.bool(False),
    # Debug flag
    debug = cms.untracked.bool(False),
    
    #define what we are importing into particle flow
    #from the various subdetectors
    # importers are executed in the order they are defined here!!!
    #order matters for some modules (it is pointed out where this is important)
    # you can find a list of all available importers in:
    #  plugins/importers
    elementImporters = cms.VPSet(
        cms.PSet( importerName = cms.string("GSFTrackImporter"),
                  source = cms.InputTag("pfTrackElec"),
                  gsfsAreSecondary = cms.bool(False),
                  superClustersArePF = cms.bool(True) ),
        cms.PSet( importerName = cms.string("ConvBremTrackImporter"),
                  source = cms.InputTag("pfTrackElec") ),
        cms.PSet( importerName = cms.string("EGPhotonImporter"),
                  source = cms.InputTag("mustachePhotons"),
                  superClustersArePF = cms.bool(True),
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
        #          gsfsAreSecondary = cms.bool(True),
        #          superClustersArePF = cms.bool(True) ),
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
    # see : plugins/kdtrees for available KDTree Types
    # to enable a KDTree for a linking pair, write a KDTree linker
    # and set useKDTree = True in the linker PSet
    #order does not matter here since we are defining a lookup table
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
        )          
)


