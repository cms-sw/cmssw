import FWCore.ParameterSet.Config as cms

offlineDQMMuonTrig = cms.EDAnalyzer("OfflineDQMMuonTrigAnalyzer",

    HltProcessName = cms.string("HLT"),

    # To disable gen or reco matching, set to an empty string
    GenLabel   = cms.untracked.string('genParticles'),
	#GenLabel   = cms.untracked.string(''),
    #RecoLabel  = cms.untracked.string('globalMuons'),

    # If the RAW trigger summary is unavailable, you can use the AOD instead
    #UseAod     = cms.untracked.bool(False),
	UseAod     = cms.untracked.bool(True),

    # These labels are not used anymore									
	AodL1Label = cms.untracked.string('hltMuLevel1PathL1Filtered'),
    AodL2Label = cms.untracked.string('hltSingleMuLevel2NoIsoL2PreFiltered'),

	# Save histograms to a separate file
	# Useful for debugging										 
	#createStandAloneHistos = cms.untracked.bool(True),
    createStandAloneHistos = cms.untracked.bool(False),
	histoFileName = cms.untracked.string("MuonTrigHistos_RECO.root"),

	# which collections will you use?

	# Using only "globalMuons" is equivalent
	# to the way relval is configured.								
	# allCollectionNames = cms.vstring ("globalMuons"),


	allCollectionNames = cms.vstring ("muons"),
									  #"barrelMuons"),
#									  "barrelMuonTracks",
#									  "overlapMuonTracks"),
									  


   customCollection = cms.VPSet(
	#cms.untracked.PSet(
	#  collectionName = cms.untracked.string ("barrelGlobalMuons"),
	#  recoCuts = cms.untracked.string ("abs(eta) < 0.9 && isStandAloneMuon() == 1"),
	#  hltCuts  = cms.untracked.string ("abs(eta) < 0.9")
	#),
	#cms.untracked.PSet(
	#  collectionName = cms.untracked.string ("overlapGlobalMuons"),
	#  recoCuts = cms.untracked.string ("abs(eta) > 0.9 && abs(eta) < 1.4 && isStandAloneMuon() == 1"),
	#  hltCuts  = cms.untracked.string ("abs(eta) > 0.9 && abs(eta) < 1.4")
	#),

	cms.untracked.PSet(
	  collectionName = cms.untracked.string ("cosmics3"),
	  trackCollection = cms.untracked.string ("globalTrack"),
	  requiredTriggers = cms.untracked.vstring(""),
	  d0cut = cms.untracked.double(10.0),
	  z0cut = cms.untracked.double(30.0),
	  recoCuts = cms.untracked.string ("pt > 3 "),
	  hltCuts  = cms.untracked.string ("pt > 3")	  
	),

	cms.untracked.PSet(
	  collectionName = cms.untracked.string ("cosmicsAna"),
	  trackCollection = cms.untracked.string ("globalTrack"),
	  requiredTriggers = cms.untracked.vstring(""),
	  d0cut = cms.untracked.double(10.0),
	  z0cut = cms.untracked.double(30.0), # 3 meters
	  recoCuts = cms.untracked.string ("pt > 10"),
	  hltCuts  = cms.untracked.string ("pt > 10")	
	),							
									  
),





#  Other possible collection names									
#									  "highPtMuonTracks"),
#									  "barrelMuonTracks",
#									  "overlapMuonTracks",
#									  "endcapMuonTracks",
#									  "externalMuonTracks"),								
											 
    # Save variables to an ntuple for more in-depth trigger studies
    NtuplePath         = cms.untracked.string('HLT_IsoMu9'),
    NtupleFileName     = cms.untracked.string(''),
    RootFileName       = cms.untracked.string(''),

    # Set the ranges and numbers of bins for histograms
    MaxPtParameters    = cms.vdouble(40,0.,80.),
    # PtParmeters is not currently used
	PtParameters       = cms.vdouble(50,0.,80.),
    EtaParameters      = cms.vdouble(50, -3.5,3.5),
    PhiParameters      = cms.vdouble(50, -3.15,3.15),
    ResParameters      = cms.vdouble(50, -0.15, 0.15),
	DrParameters       = cms.vdouble(50, 0.0, 0.05),

    # Set cuts placed on the generated muons and matching criteria
    # Use pt cut just below 10 to allow through SingleMuPt10 muons
	# All of this is obselete								
    MinPtCut           = cms.untracked.double(0),
    MaxEtaCut          = cms.untracked.double(3.5),
    MotherParticleId   = cms.untracked.uint32(0),
   # L1DrCut            = cms.untracked.double(0.4),
   # L2DrCut            = cms.untracked.double(0.25),
   # L3DrCut            = cms.untracked.double(0.015),
	# for testing

	matchType = cms.untracked.string("cosmic"),
    # If you have cosmic matching
	# you will ignore the delta R cuts								
									
	L1DrCut   = cms.untracked.double(0.5),
	L2DrCut   = cms.untracked.double(0.5),
	L3DrCut   = cms.untracked.double(0.5),								
									
	#L3DrCut            = cms.untracked.double(0.1),
    TriggerResultLabel = cms.InputTag("TriggerResults","","HLT"),

    DQMStore = cms.untracked.bool(True),
    CrossSection = cms.double(0.97),
    NSigmas90 = cms.untracked.vdouble(3.0, 3.0, 3.0, 3.0),

    TriggerNames = cms.vstring(
        "HLT_L1Mu",
        "HLT_L1MuOpen",
		"HLT_L2Mu3",
		"HLT_OIstateTkMu3",
		"HLT_TrackerCosmics_CTF",
    ),


    # This timing module is deprecated
    TimerLabel = cms.InputTag("hltTimer"),
    TimeNbins = cms.uint32(150),
    MaxTime = cms.double(150.0),
    TimingModules = cms.untracked.PSet(
        MuonL3IsoModules = cms.untracked.vstring('pixelTracks', 
            'hltL3MuonIsolations', 
            'SingleMuIsoL3IsoFiltered'),
        TrackerRecModules = cms.untracked.vstring('siPixelClusters', 
            'siPixelRecHits', 
            'siStripClusters', 
            'siStripMatchedRecHits'),
        MuonLocalRecModules = cms.untracked.vstring('dt1DRecHits', 
            'dt4DSegments', 
            'rpcRecHits', 
            'csc2DRecHits', 
            'cscSegments'),
        CaloDigiModules = cms.untracked.vstring('ecalDigis', 
            'ecalPreshowerDigis', 
            'hcalDigis'),
        MuonL3RecModules = cms.untracked.vstring('hltL3Muons', 
            'hltL3MuonCandidates', 
            'SingleMuIsoL3PreFiltered'),
        CaloRecModules = cms.untracked.vstring('ecalRegionalMuonsWeightUncalibRecHit', 
            'ecalRegionalMuonsRecHit', 
            'ecalPreshowerRecHit', 
            'hbhereco', 
            'horeco', 
            'hfreco', 
            'towerMakerForMuons', 
            'caloTowersForMuons'),
        MuonL2IsoModules = cms.untracked.vstring('hltL2MuonIsolations', 
            'SingleMuIsoL2IsoFiltered'),
        MuonDigiModules = cms.untracked.vstring('muonCSCDigis', 
            'muonDTDigis', 
            'muonRPCDigis'),
        TrackerDigiModules = cms.untracked.vstring('siPixelDigis', 
            'siStripDigis'),
        MuonL2RecModules = cms.untracked.vstring('hltL2MuonSeeds', 
            'hltL2Muons', 
            'hltL2MuonCandidates', 
            'SingleMuIsoL2PreFiltered')
    )
)
