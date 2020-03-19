import FWCore.ParameterSet.Config as cms

particleFlowBlock = cms.EDProducer(
    "PFBlockProducer",
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
        cms.PSet( importerName = cms.string("SuperClusterImporter"),
                  source_eb = cms.InputTag("particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel"),
                  source_ee = cms.InputTag("particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower"),
                  source_towers = cms.InputTag("towerMaker"),
                  maximumHoverE = cms.double(0.5),
                  minSuperClusterPt = cms.double(10.0),
                  minPTforBypass = cms.double(100.0),
                  superClustersArePF = cms.bool(True) ),
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
		  trackQuality = cms.string("highPurity"),
                  cleanBadConvertedBrems = cms.bool(True),
                  useIterativeTracking = cms.bool(True),
                  DPtOverPtCuts_byTrackAlgo = cms.vdouble(10.0,10.0,10.0,
                                                           10.0,10.0,5.0),
                  NHitCuts_byTrackAlgo = cms.vuint32(3,3,3,3,3,3)
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
                  source = cms.InputTag("particleFlowBadHcalPseudoCluster") ),
        cms.PSet( importerName = cms.string("GenericClusterImporter"),
                  source = cms.InputTag("particleFlowClusterHO") ),
        cms.PSet( importerName = cms.string("GenericClusterImporter"),
                  source = cms.InputTag("particleFlowClusterHF") ),
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
                  useKDTree  = cms.bool(True),
                  trajectoryLayerEntrance = cms.string("HCALEntrance"),
                  trajectoryLayerExit = cms.string("HCALExit")),
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
        cms.PSet( linkerName = cms.string("GSFAndBREMLinker"),
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

for imp in particleFlowBlock.elementImporters:
  if imp.importerName.value() == "SuperClusterImporter":
    _scImporter = imp

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(_scImporter,
                                minSuperClusterPt = 1.0,
                                minPTforBypass = 0.0)

def _findIndicesByModule(name):
   ret = []
   for i, pset in enumerate(particleFlowBlock.elementImporters):
        if pset.importerName.value() == name:
            ret.append(i)
   return ret


from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
# kill tracks in the HGCal
_insertGeneralTracksImporter = {}
for idx in _findIndicesByModule('GeneralTracksImporter'):
    _insertGeneralTracksImporter[idx] = dict(
        importerName = cms.string('GeneralTracksImporterWithVeto'),
        veto = cms.InputTag('hgcalTrackCollection:TracksInHGCal')
    )
phase2_hgcal.toModify(
    particleFlowBlock,
    elementImporters = _insertGeneralTracksImporter
)
### for later
#_phase2_hgcal_Linkers.append(
#    cms.PSet( linkerName = cms.string("SCAndHGCalLinker"),
#              linkType   = cms.string("SC:HGCAL"),
#              useKDTree  = cms.bool(False),
#              SuperClusterMatchByRef = cms.bool(True) )
#)
#_phase2_hgcal_Linkers.append(
#    cms.PSet( linkerName = cms.string("HGCalAndBREMLinker"),
#              linkType   = cms.string("HGCAL:BREM"),
#              useKDTree  = cms.bool(False) )
#)
#_phase2_hgcal_Linkers.append(
#    cms.PSet( linkerName = cms.string("GSFAndHGCalLinker"),
#                  linkType   = cms.string("GSF:HGCAL"),
#                  useKDTree  = cms.bool(False) )
#)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
_addTrackHFLinks = particleFlowBlock.linkDefinitions.copy()
_addTrackHFLinks.append(
  cms.PSet( linkerName = cms.string("TrackAndHCALLinker"),
            linkType   = cms.string("TRACK:HFEM"),
            useKDTree  = cms.bool(True),
            trajectoryLayerEntrance = cms.string("VFcalEntrance"),
            trajectoryLayerExit = cms.string(""))
  )
_addTrackHFLinks.append(
  cms.PSet( linkerName = cms.string("TrackAndHCALLinker"),
            linkType   = cms.string("TRACK:HFHAD"),
            useKDTree  = cms.bool(True),
            trajectoryLayerEntrance = cms.string("VFcalEntrance"),
            trajectoryLayerExit = cms.string(""))
)
phase2_tracker.toModify(
    particleFlowBlock,
    linkDefinitions = _addTrackHFLinks
)

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
_addTiming = particleFlowBlock.elementImporters.copy()
_addTiming.append( cms.PSet( importerName = cms.string("TrackTimingImporter"),
                             timeValueMap = cms.InputTag("trackTimeValueMapProducer:generalTracksConfigurableFlatResolutionModel"),
                             timeErrorMap = cms.InputTag("trackTimeValueMapProducer:generalTracksConfigurableFlatResolutionModelResolution"),
                             timeValueMapGsf = cms.InputTag("gsfTrackTimeValueMapProducer:electronGsfTracksConfigurableFlatResolutionModel"),
                             timeErrorMapGsf = cms.InputTag("gsfTrackTimeValueMapProducer:electronGsfTracksConfigurableFlatResolutionModelResolution")
                             )
                   )

from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
_addTimingLayer = particleFlowBlock.elementImporters.copy()
_addTimingLayer.append( cms.PSet( importerName = cms.string("TrackTimingImporter"),
                             timeValueMap = cms.InputTag("tofPID:t0"),
                             timeErrorMap = cms.InputTag("tofPID:sigmat0"),
                             #this will cause no time to be set for gsf tracks
                             #(since this is not available for the fullsim/reconstruction yet)
                             #*TODO* update when gsf times are available
                             timeValueMapGsf = cms.InputTag("tofPID:t0"),
                             timeErrorMapGsf = cms.InputTag("tofPID:sigmat0")
                             )
                   )

phase2_timing.toModify(
    particleFlowBlock,
    elementImporters = _addTiming
)

phase2_timing_layer.toModify(
    particleFlowBlock,
    elementImporters = _addTimingLayer
)
