import FWCore.ParameterSet.Config as cms

l1tTrackJetsEmulation = cms.EDProducer('L1TrackJetEmulatorProducer',
        L1TrackInputTag= cms.InputTag("l1tTrackVertexAssociationProducerForJets", "Level1TTTracksSelectedAssociatedEmulation"),
        L1PVertexInputTag=cms.InputTag("l1tVertexFinderEmulator","L1VerticesEmulation"),
        MaxDzTrackPV = cms.double(1.0),
        trk_zMax = cms.double (15.) ,    # maximum track z
	trk_ptMax = cms.double(200.),    # maximumum track pT before saturation [GeV]
	trk_ptMin = cms.double(2.0),     # minimum track pt [GeV]
   	trk_etaMax = cms.double(2.4),    # maximum track eta
        nStubs4PromptChi2=cms.double(10.0), #Prompt track quality flags for loose/tight
        nStubs4PromptBend=cms.double(2.2),
        nStubs5PromptChi2=cms.double(10.0),
        nStubs5PromptBend=cms.double(2.2),
	trk_nPSStubMin=cms.int32(-1),    # minimum PS stubs, -1 means no cut
	minTrkJetpT=cms.double(-1.),      # minimum track pt to be considered for track jet
	etaBins=cms.int32(24),
	phiBins=cms.int32(27),
	zBins=cms.int32(1),
        d0_cutNStubs4=cms.double(-1),
        d0_cutNStubs5=cms.double(-1),
	lowpTJetMinTrackMultiplicity=cms.int32(2),
        lowpTJetThreshold=cms.double(50.),
	highpTJetMinTrackMultiplicity=cms.int32(3),
        highpTJetThreshold=cms.double(100.),
	displaced=cms.bool(False), #Flag for displaced tracks
	nStubs4DisplacedChi2=cms.double(5.0), #Displaced track quality flags for loose/tight
	nStubs4DisplacedBend=cms.double(1.7),
	nStubs5DisplacedChi2=cms.double(2.75),
	nStubs5DisplacedBend=cms.double(3.5),
	nDisplacedTracks=cms.int32(2) #Number of displaced tracks required per jet
)

l1tTrackJetsExtendedEmulation = l1tTrackJetsEmulation.clone(
	L1TrackInputTag= cms.InputTag("l1tTrackVertexAssociationProducerExtendedForJets", "Level1TTTracksExtendedSelectedAssociatedEmulation"),
        L1PVertexInputTag=cms.InputTag("l1tVertexFinderEmulator", "L1VerticesEmulation"),
	minTrkJetpT= 5.0,      # minimum track pt to be considered for track jet
        MaxDzTrackPV = 5.0,
	d0_cutNStubs4= -1, # -1 excludes nstub=4 from disp tag
	d0_cutNStubs5= 0.22,
	displaced= True, #Flag for displaced tracks
	nStubs4DisplacedChi2= 3.3, #Disp tracks selection [trk<cut]
	nStubs4DisplacedBend= 2.3,
	nStubs5DisplacedChi2= 11.3,
	nStubs5DisplacedBend= 9.8,
	nDisplacedTracks= 3 #min Ntracks to tag a jet as displaced
)
