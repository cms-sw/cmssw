import FWCore.ParameterSet.Config as cms

l1tTrackJetsEmulation = cms.EDProducer('L1TrackJetEmulatorProducer',
        L1TrackInputTag= cms.InputTag("l1tTrackVertexAssociationProducerForJets", "Level1TTTracksSelectedAssociatedEmulation"),
        trk_zMax = cms.double (15.) ,    # maximum track z
	trk_ptMax = cms.double(200.),    # maximumum track pT before saturation [GeV]
   	trk_etaMax = cms.double(2.4),    # maximum track eta
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
	nDisplacedTracks=cms.int32(2) #Number of displaced tracks required per jet
)

l1tTrackJetsExtendedEmulation = l1tTrackJetsEmulation.clone(
	L1TrackInputTag= cms.InputTag("l1tTrackVertexAssociationProducerExtendedForJets", "Level1TTTracksExtendedSelectedAssociatedEmulation"),
	minTrkJetpT= 5.0,      # minimum track pt to be considered for track jet
	d0_cutNStubs4= -1, # -1 excludes nstub=4 from disp tag
	d0_cutNStubs5= 0.22,
	displaced= True, #Flag for displaced tracks
	nDisplacedTracks= 3 #min Ntracks to tag a jet as displaced
)
