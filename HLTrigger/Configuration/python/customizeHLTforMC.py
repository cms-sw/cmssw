import FWCore.ParameterSet.Config as cms

def customizeHLTforFastSim(process,_fastSim=True):
  process=customizeHLTforMC(process,_fastSim)
  return process

def customizeHLTforFullSim(process,_fastSim=False):
  process=customizeHLTforMC(process,_fastSim)
  return process

def customizeHLTforMC(process,_fastSim=False):
  """adapt the HLT to run on MC, instead of data
  see Configuration/StandardSequences/Reconstruction_Data_cff.py
  which does the opposite, for RECO"""

  # PFRecHitProducerHCAL
  if 'hltParticleFlowRecHitHCAL' in process.__dict__:
    process.hltParticleFlowRecHitHCAL.ApplyPulseDPG      = cms.bool(False)
    process.hltParticleFlowRecHitHCAL.LongShortFibre_Cut = cms.double(1000000000.0)

  # customise hltHbhereco to use the Method 3 time slew parametrization and response correction for Monte Carlo (PR #11091)
  if 'hltHbhereco' in process.__dict__:
    process.hltHbhereco.pedestalSubtractionType = cms.int32( 1 )
    process.hltHbhereco.pedestalUpperLimit      = cms.double( 2.7 ) 
    process.hltHbhereco.timeSlewParsType        = cms.int32( 3 )
    #old MC
    process.hltHbhereco.timeSlewPars            = cms.vdouble( 9.27638, -2.05585, 0, 9.27638, -2.05585, 0, 9.27638, -2.05585, 0 ) 
    process.hltHbhereco.respCorrM3              = cms.double( 1.0 )
    #new MC once completely implemented
#    process.hltHbhereco.timeSlewPars            = cms.vdouble( 12.2999, -2.19142, 0, 12.2999, -2.19142, 0, 12.2999, -2.19142, 0 )
#    process.hltHbhereco.respCorrM3              = cms.double( 0.95 )


  if _fastSim:

    fastsim = cms.ProcessFragment( process.name_() )
    fastsim.load( "FastSimulation.HighLevelTrigger.HLTSetup_cff" )

    fastSimUnsupportedPaths = (

      # paths for which a recovery is not foreseen/possible
      "AlCa_*_v*",
      "DQM_*_v*",
      "HLT_*Calibration_v*",
      "HLT_DTErrors_v*",
      "HLT_Random_v*",
      "HLT_HcalNZS_v*",
      "HLT_HcalPhiSym_v*",
      "HLT_Activity_Ecal*_v*",
      "HLT_IsoTrackHB_v*",
      "HLT_IsoTrackHE_v*",
      "HLT_L1SingleMuOpen_AntiBPTX_v*",
      "HLT_JetE*_NoBPTX*_v*",
      "HLT_L2Mu*_NoBPTX*_v*",
      "HLT_Beam*_v*",
      #"HLT_L1Tech_*_v*",
      "HLT_HI*",
      "HLT_GlobalRunHPDNoise_v*",
      "HLT_L1TrackerCosmics_v*",
      "HLT_HcalUTCA_v*",

      # TODO: paths not supported by FastSim, but for which a recovery should be attempted
      "HLT_DoubleMu33NoFiltersNoVtx_v*",
      "HLT_DoubleMu38NoFiltersNoVtx_v*",
      "HLT_Mu38NoFiltersNoVtx_Photon38_CaloIdL_v*",
      "HLT_Mu42NoFiltersNoVtx_Photon42_CaloIdL_v*",
      "HLT_DoubleMu23NoFiltersNoVtxDisplaced_v*",
      "HLT_DoubleMu28NoFiltersNoVtxDisplaced_v*",
      "HLT_Mu28NoFiltersNoVtxDisplaced_Photon28_CaloIdL_v*",
      "HLT_Mu33NoFiltersNoVtxDisplaced_Photon33_CaloIdL_v*",
      "HLT_HT350_DisplacedDijet80_Tight_DisplacedTrack_v*",
      "HLT_HT350_DisplacedDijet80_DisplacedTrack_v*",
      "HLT_HT500_DisplacedDijet40_Inclusive_v*",
      "HLT_HT350_DisplacedDijet40_DisplacedTrack_v*",
      "HLT_HT550_DisplacedDijet40_Inclusive_v*",
      "HLT_HT350_DisplacedDijet80_DisplacedTrack_v*",
      "HLT_HT400_DisplacedDijet40_Inclusive_v*",
      "HLT_HT250_DisplacedDijet40_DisplacedTrack_v*",
      "HLT_TrkMu15_DoubleTrkMu5NoFiltersNoVtx_v*",
      "HLT_TrkMu17_DoubleTrkMu8NoFiltersNoVtx_v*",
      "HLT_MET60_IsoTrk*",
      "HLT_MET75_IsoTrk50_v*",
      "HLT_MET90_IsoTrk50_v*",
      "HLT_VBF_DisplacedJet40_DisplacedTrack_v*",
      "HLT_VBF_DisplacedJet40_TightID_DisplacedTrack_v*",
      "HLT_VBF_DisplacedJet40_VTightID_DisplacedTrack_v*",
      "HLT_VBF_DisplacedJet40_VVTightID_DisplacedTrack_v*",
      "HLT_VBF_DisplacedJet40_Hadronic_2PromptTrack_v*",
      "HLT_VBF_DisplacedJet40_DisplacedTrack_2TrackIP2DSig5_v*",
      "HLT_Mu33NoFiltersNoVtxDisplaced_DisplacedJet50_Tight_v*",
      "HLT_Mu33NoFiltersNoVtxDisplaced_DisplacedJet50_Loose_v*",
      "HLT_Mu38NoFiltersNoVtxDisplaced_DisplacedJet60_Tight_v*",
      "HLT_Mu38NoFiltersNoVtxDisplaced_DisplacedJet60_Loose_v*",
      "HLT_Mu38NoFiltersNoVtx_DisplacedJet60_Loose_v*",
      "HLT_Mu28NoFiltersNoVtx_DisplacedJet40_Loose_v*",
      "HLT_Mu28NoFiltersNoVtx_CentralCaloJet40_v*",
      "HLT_Mu23NoFiltersNoVtx_Photon23_CaloIdL_v*",
      "HLT_DoubleMu18NoFiltersNoVtx_v*",
      "HLT_DoubleMuNoFiltersNoVtx_SaveObjects_v*",
      "MC_DoubleMuNoFiltersNoVtx_v*",
      "HLT_L1MuOpenNotHF2Pixel_SingleTrack*",
      "HLT_L1TOTEM0_RomanPotsAND_PixelClusters*",
      )

    ESModulesToRemove = (
#      "CaloTowerGeometryFromDBEP",
#      "CastorGeometryFromDBEP",
#      "EcalBarrelGeometryFromDBEP",
#      "EcalEndcapGeometryFromDBEP",
#      "EcalPreshowerGeometryFromDBEP",
#      "HcalGeometryFromDBEP",
#      "ZdcGeometryFromDBEP",
#      "XMLFromDBSource",
#      "sistripconn",

#      "navigationSchoolESProducer",
#      "TransientTrackBuilderESProducer",
#      "SteppingHelixPropagatorAny",
#      "OppositeMaterialPropagator",
#      "MaterialPropagator",
#      "CaloTowerConstituentsMapBuilder",
#      "CaloTopologyBuilder",
      )

    ModulesToRemove = (
      #   "hltL3MuonIsolations",
      #   "hltPixelVertices",
      "hltCkfL1SeededTrackCandidates",
      "hltCtfL1SeededithMaterialTracks",
      "hltCkf3HitL1SeededTrackCandidates",
      "hltCtf3HitL1SeededWithMaterialTracks",
      "hltCkf3HitActivityTrackCandidates",
      "hltCtf3HitActivityWithMaterialTracks",
      "hltActivityCkfTrackCandidatesForGSF",
      "hltL1SeededCkfTrackCandidatesForGSF",
      "hltMuCkfTrackCandidates",
      "hltMuCtfTracks",
      "hltTau3MuCkfTrackCandidates",
      "hltTau3MuCtfWithMaterialTracks",
      "hltMuTrackJpsiCkfTrackCandidates",
      "hltMuTrackJpsiCtfTracks",
      "hltMuTrackJpsiEffCkfTrackCandidates",
      "hltMuTrackJpsiEffCtfTracks",
      "hltJpsiTkPixelSeedFromL3Candidate",
      "hltCkfTrackCandidatesJpsiTk",
      "hltCtfWithMaterialTracksJpsiTk",
      "hltMuTrackCkfTrackCandidatesOnia",
      "hltMuTrackCtfTracksOnia",

      "hltFEDSelector",
      "hltL3TrajSeedOIHit",
      "hltL3TrajSeedIOHit",
      "hltL3NoFiltersTrajSeedOIHit",
      "hltL3NoFiltersTrajSeedIOHit",
      "hltL3TrackCandidateFromL2OIState",
      "hltL3TrackCandidateFromL2OIHit",
      "hltL3TrackCandidateFromL2IOHit",
      "hltL3TrackCandidateFromL2NoVtx",
      "hltHcalDigis",
      "hltHoreco",
      "hltHfreco",
      "hltHbhereco",
      "hltESRawToRecHitFacility",
      "hltEcalRecHitAll",
      "hltESRecHitAll",
      # === eGamma
      "hltEgammaCkfTrackCandidatesForGSF",
      "hltEgammaGsfTracks",
      "hltEgammaCkfTrackCandidatesForGSFUnseeded",
      "hltEgammaGsfTracksUnseeded",
      # === hltPF
      "hltPFJetCkfTrackCandidates",
      "hltPFJetCtfWithMaterialTracks",
      "hltPFlowTrackSelectionHighPurity",
      # === hltFastJet
      "hltDisplacedHT250L1FastJetRegionalPixelSeedGenerator",
      "hltDisplacedHT250L1FastJetRegionalCkfTrackCandidates",
      "hltDisplacedHT250L1FastJetRegionalCtfWithMaterialTracks",
      "hltDisplacedHT300L1FastJetRegionalPixelSeedGenerator",
      "hltDisplacedHT300L1FastJetRegionalCkfTrackCandidates",
      "hltDisplacedHT300L1FastJetRegionalCtfWithMaterialTracks",
      "hltBLifetimeRegionalPixelSeedGeneratorbbPhiL1FastJet",
      "hltBLifetimeRegionalCkfTrackCandidatesbbPhiL1FastJet",
      "hltBLifetimeRegionalCtfWithMaterialTracksbbPhiL1FastJet",
      "hltBLifetimeRegionalPixelSeedGeneratorHbbVBF",
      "hltBLifetimeRegionalCkfTrackCandidatesHbbVBF",
      "hltBLifetimeRegionalCtfWithMaterialTracksHbbVBF",
      "hltBLifetimeBTagIP3D1stTrkRegionalPixelSeedGeneratorJet20HbbL1FastJet",
      "hltBLifetimeBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20HbbL1FastJet",
      "hltBLifetimeBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20HbbL1FastJet",
      "hltBLifetimeDiBTagIP3D1stTrkRegionalPixelSeedGeneratorJet20HbbL1FastJet",
      "hltBLifetimeDiBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20HbbL1FastJet",
      "hltBLifetimeDiBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20HbbL1FastJet",
      # === hltBLifetimeRegional
      "hltBLifetimeRegionalPixelSeedGeneratorHbb",
      "hltBLifetimeRegionalCkfTrackCandidatesHbb",
      "hltBLifetimeRegionalCtfWithMaterialTracksHbb",
      "hltBLifetimeRegionalPixelSeedGeneratorbbPhi",
      "hltBLifetimeRegionalCkfTrackCandidatesbbPhi",
      "hltBLifetimeRegionalCtfWithMaterialTracksbbPhi",
      "hltBLifetimeBTagIP3D1stTrkRegionalPixelSeedGeneratorJet20Hbb",
      "hltBLifetimeBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20Hbb",
      "hltBLifetimeBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20Hbb",
      "hltBLifetimeDiBTagIP3D1stTrkRegionalPixelSeedGeneratorJet20Hbb",
      "hltBLifetimeDiBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20Hbb",
      "hltBLifetimeDiBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20Hbb",
      "hltBLifetimeFastRegionalPixelSeedGeneratorHbbVBF",
      "hltBLifetimeFastRegionalCkfTrackCandidatesHbbVBF",
      "hltBLifetimeFastRegionalCtfWithMaterialTracksHbbVBF",
      "hltBLifetimeRegionalPixelSeedGeneratorbbPhiL1FastJetFastPV",
      "hltBLifetimeRegionalCkfTrackCandidatesbbPhiL1FastJetFastPV",
      "hltBLifetimeRegionalCtfWithMaterialTracksbbPhiL1FastJetFastPV",
      "hltFastPixelBLifetimeRegionalPixelSeedGeneratorHbb",
      "hltFastPixelBLifetimeRegionalCkfTrackCandidatesHbb",
      "hltFastPixelBLifetimeRegionalCtfWithMaterialTracksHbb",

      "hltPixelTracksForMinBias",
      "hltPixelTracksForMinBias01",
      "hltPixelTracksForHighMult",
      "hltRegionalPixelTracks",
      "hltPixelTracksReg",
      "hltPixelTracksL3Muon",
      "hltPixelTracksGlbTrkMuon",
      "hltPixelTracksHighPtTkMuIso",
      "hltPixelTracksHybrid",
      "hltPixelTracksForPhotons",
      "hltPixelTracksForEgamma",
      "hltPixelTracksElectrons",
      "hltPixelTracksForHighPt",
      "hltHighPtPixelTracks",
      "hltPixelTracksForNoPU",

      "hltFastPixelHitsVertex",
      "hltFastPixelTracks",
      "hltFastPixelTracksRecover",

      "hltPixelLayerPairs",
      "hltPixelLayerTriplets",
      "hltPixelLayerTripletsReg",
      "hltPixelLayerTripletsHITHB",
      "hltPixelLayerTripletsHITHE",
      "hltMixedLayerPairs",

      "hltFastPrimaryVertexbbPhi",
      "hltPixelTracksFastPVbbPhi",
      "hltPixelTracksRecoverbbPhi",
      "hltFastPixelHitsVertexVHbb",
      "hltFastPixelTracksVHbb",
      "hltFastPixelTracksRecoverVHbb",

      "hltFastPrimaryVertex",
      "hltFastPVPixelVertexFilter",
      "hltFastPVPixelTracks",
      "hltFastPVPixelTracksRecover",

      #   "hltPixelMatchElectronsActivity",

      "hltMuonCSCDigis",
      "hltMuonDTDigis",
      "hltMuonRPCDigis",
      "hltGtDigis",
#      "hltL1GtTrigReport",
      #   "hltCsc2DRecHits",
      #   "hltDt1DRecHits",
      #   "hltRpcRecHits",
      "hltScalersRawToDigi",
      "hltEcalPreshowerDigis",
      "hltEcalDigis",
      "hltEcalDetIdToBeRecovered",

      )

    SequencesToRemove = (
      "HLTL1SeededEgammaRegionalRecoTrackerSequence",
      "HLTEcalActivityEgammaRegionalRecoTrackerSequence",
      "HLTPixelMatchElectronActivityTrackingSequence",
      "HLTDoLocalStripSequence",
      "HLTDoLocalPixelSequence",
      "HLTDoLocalPixelSequenceRegL2Tau",
      "HLTDoLocalStripSequenceReg",
      "HLTDoLocalPixelSequenceReg",
      "HLTDoLocalStripSequenceRegForBTag",
      "HLTDoLocalPixelSequenceRegForBTag",
      "HLTDoLocalPixelSequenceRegForNoPU",
      #   "hltSiPixelDigis",
      #   "hltSiPixelClusters",
      #   "hltSiPixelRecHits",
      "HLTRecopixelvertexingSequence",
#      "HLTEndSequence",
      "HLTBeginSequence",
      "HLTBeginSequenceNZS",
      "HLTBeginSequenceBPTX",
      "HLTBeginSequenceAntiBPTX",
      "HLTHBHENoiseSequence",
      "HLTIterativeTrackingIter04",
      "HLTIterativeTrackingIter02",
      "HLTIterativeTracking",
      "HLTIterativeTrackingForHighPt",
      "HLTIterativeTrackingTau3Mu",
      "HLTIterativeTrackingReg",
      "HLTIterativeTrackingForPA",
      "HLTIterativeTrackingForElectronIter02",
      "HLTIterativeTrackingForPhotonsIter02",
      "HLTIterativeTrackingL3MuonIter02",
      "HLTIterativeTrackingGlbTrkMuonIter02",
      "HLTIterativeTrackingL3MuonRegIter02",
      "HLTIterativeTrackingHighPtTkMu",
      "HLTIterativeTrackingHighPtTkMuIsoIter02",
      "HLTIterativeTrackingForBTagIter02",
      "HLTIterativeTrackingForBTagIter12",
      "HLTIterativeTrackingForTauIter04",
      "HLTIterativeTrackingForTauIter02",
      "HLTIterativeTrackingDisplacedJpsiIter02",
      "HLTIterativeTrackingDisplacedPsiPrimeIter02",
      "HLTIterativeTrackingDisplacedNRMuMuIter02",
      "HLTIterativeTrackingForBTagIteration0",
      "HLTIterativeTrackingIteration4DisplacedJets",
      "HLTRegionalCKFTracksForL3Isolation",
      "HLTHBHENoiseCleanerSequence",
      )

# Removing ESmodules
    for label in ESModulesToRemove:
      if (hasattr(process,label)):
        delattr(process,label)

# Removing paths
    import fnmatch,re
    ExplicitList = []
    HLTSchedule = tuple( path.label_() for path in process.HLTSchedule)
    for black in fastSimUnsupportedPaths:
      compiled = re.compile(fnmatch.translate(black))
      for path in HLTSchedule:
        if compiled.search(path) is not None:
          ExplicitList += [path]
    UniqueList = []
    for path in ExplicitList:
      if path not in UniqueList:
        UniqueList += [path]
    for path in UniqueList:
      process.schedule.remove(getattr(process,path))
    process.setSchedule_(process.schedule)
    process.prune()

# Removing streams and datasets PSets
    if hasattr(process,'streams'):
      delattr(process,'streams')
    if hasattr(process,'datasets'):
      delattr(process,'datasets')

# Removing sequences, possibly to be taken from fastsim import
    for label in SequencesToRemove:
      if hasattr(process,label):
        if hasattr(fastsim,label):
          setattr(process,label,getattr(fastsim,label))
        else:
          object = getattr(process,label)
          list = tuple(process.sequences_().keys())
          for name in list:
            sequence = getattr(process,name)
            more=True
            while more:
              more = sequence.remove(object)
          list = tuple(process.paths_().keys())
          for name in list:
            path = getattr(process,name)
            more=True
            while more:
              more = path.remove(object)
          delattr(process,label)

# Removing modules, possibly to be taken from fastsim import
    for label in ModulesToRemove:
      if hasattr(process,label):
        if hasattr(fastsim,label):
          setattr(process,label,getattr(fastsim,label))
        else:
          object = getattr(process,label)
          list = tuple(process.sequences_().keys())
          for name in list:
            sequence = getattr(process,name)
            more=True
            while more:
              more = sequence.remove(object)
          list = tuple(process.paths_().keys())
          for name in list:
            path = getattr(process,name)
            more=True
            while more:
              more = path.remove(object)
          delattr(process,label)

# Special transformations:
    if hasattr(process,'hltGetConditions') and hasattr(process,'HLTriggerFirstPath'):
      process.hltDummyConditions = cms.EDFilter( "HLTBool", result = cms.bool( True ) )
      process.HLTriggerFirstPath.replace(process.hltGetConditions,process.hltDummyConditions)
    if hasattr(process,'hltPixelVertices'):
      process.hltPixelVertices.beamSpot = cms.InputTag('offlineBeamSpot')
    specialModules = ( 'hltEcalRecHit', 'hltEcalPreshowerRecHit', )
    for label in specialModules:
      if hasattr(process,label):
        setattr(fastsim,label,getattr(process,label))

# Use fastsim imports
    for label in fastsim.producers_().keys():
      if hasattr(process,label):
        setattr(process,label,getattr(fastsim,label))
    for label in fastsim.filters_().keys():
      if hasattr(process,label):
        setattr(process,label,getattr(fastsim,label))
    for label in fastsim.analyzers_().keys():
      if hasattr(process,label):
        setattr(process,label,getattr(fastsim,label))
    for label in fastsim.sequences_().keys():
      if hasattr(process,label):
        setattr(process,label,getattr(fastsim,label))

# Update InputTags

    InputTags = (
      ('hltGtDigis','gtDigis'),
      ('hltL1GtObjectMap','gtDigis'),
      ('hltEcalDigis:ebDigis','ecalDigis:ebDigis'),
      ('hltEcalDigis:eeDigis','ecalDigis:eeDigis'),
      ('hltMuonCSCDigis','muonCSCDigis'),
      ('hltMuonCSCDigis:MuonCSCStripDigi','muonCSCDigis:MuonCSCStripDigi'),
      ('hltMuonCSCDigis:MuonCSCWireDigi','muonCSCDigis:MuonCSCWireDigi'),
      ('hltMuonDTDigis','muonDTDigis'),
      ('hltMuonRPCDigis','muonRPCDigis'),
      ('hltEcalPreshowerDigis','ecalPreshowerDigis'),
      ('hltHbhereco', 'hbhereco'),
      ('hltHoreco', 'horeco'),
      ('hltHfreco', 'hfreco'),

      ('hltIter2Merged', 'generalTracks'),
      ('hltIter2HighPtMerged', 'generalTracks'),
      ('hltIter2MergedForElectrons', 'generalTracks'),
      ('hltIter2MergedForPhotons', 'generalTracks'),
      ('hltIter2L3MuonMerged', 'generalTracks'),
      ('hltIter2MergedForBTag', 'generalTracks'),
      ('hltIter2MergedForTau', 'generalTracks'),
      ('hltIter2GlbTrkMuonMerged', 'generalTracks'),
      ('hltIter2HighPtTkMuMerged', 'generalTracks'),
      ('hltIter2HighPtTkMuIsoMerged', 'generalTracks'),
      ('hltIter2DisplacedJpsiMerged', 'generalTracks'),
      ('hltIter2DisplacedPsiPrimeMerged', 'generalTracks'),
      ('hltIter2DisplacedNRMuMuMerged', 'generalTracks'),
      ('hltIter0PFlowTrackSelectionHighPurityForBTag', 'generalTracks'),
      ('hltIter4HighPtMerged', 'generalTracks'),
      ('hltIterativeTrackingForPAMerged', 'generalTracks'),

      ('hltRegionalTracksForL3MuonIsolation', 'hltPixelTracks'),

      ('hltL1extraParticles','l1extraParticles'),
      ('hltL1extraParticles:Central','l1extraParticles:Central'),
      ('hltL1extraParticles:Forward','l1extraParticles:Forward'),
      ('hltL1extraParticles:Isolated','l1extraParticles:Isolated'),
      ('hltL1extraParticles:NonIsolated','l1extraParticles:NonIsolated'),
      ('hltL1extraParticles:Tau','l1extraParticles:Tau'),
      ('hltL1extraParticles:IsoTau','l1extraParticles:IsoTau'),
      ('hltOfflineBeamSpot','offlineBeamSpot'),
      ('hltOnlineBeamSpot','offlineBeamSpot'),
      ('hltSiStripClusters','MeasurementTrackerEvent'),

      )
    from HLTrigger.Configuration.CustomConfigs import MassReplaceInputTag
    for pair in InputTags:
      process = MassReplaceInputTag(process,pair[0],pair[1])

# Update top-level named parameters
    NamedParameters = (
      ('GMTReadoutCollection',cms.InputTag('gtDigis'),cms.InputTag('gmtDigis')),
      ('killDeadChannels',cms.bool(True),cms.bool(False)),
      ('recoverEBFE',cms.bool(True),cms.bool(False)),
      ('recoverEEFE',cms.bool(True),cms.bool(False)),
      ('src',cms.InputTag('hltHcalTowerNoiseCleaner'),cms.InputTag('hltTowerMakerForAll')),
      ('initialSeeds',cms.InputTag('noSeedsHere'),cms.InputTag('globalPixelSeeds')),
      ('preFilteredSeeds',cms.bool(True),cms.bool(False)),
      )
    from HLTrigger.Configuration.CustomConfigs import MassReplaceParameter
    for thrice in NamedParameters:
      process = MassReplaceParameter(process,thrice[0],thrice[1],thrice[2])

# Update nested named parameters
    for label in ('hltEgammaElectronPixelSeeds','hltEgammaElectronPixelSeedsUnseeded',):
      if hasattr(process,label):
        getattr(process,label).SeedConfiguration.initialSeeds = cms.InputTag('globalPixelSeeds')
        getattr(process,label).SeedConfiguration.preFilteredSeeds = cms.bool(False)
        if hasattr(fastsim,'globalPixelSeeds'): 
          if hasattr(fastsim.globalPixelSeeds,'outputSeedCollectionName'):
            getattr(process,label).SeedConfiguration.initialSeeds = cms.InputTag('globalPixelSeeds',fastsim.globalPixelSeeds.outputSeedCollectionName.value())


# Extending fastsim import
    fastsim.extend(process)
    fastsim.setSchedule_(fastsim.schedule)
    fastsim.prune()

    return fastsim

  else:

    return process
