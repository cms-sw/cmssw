import FWCore.ParameterSet.Config as cms

#prompt jet selection
l1tTrackJets = cms.EDProducer('L1TrackJetProducer',
        L1TrackInputTag = cms.InputTag("l1tTrackVertexAssociationProducerForJets", "Level1TTTracksSelectedAssociated"),
	trk_zMax = cms.double (15.) ,    # maximum track z
	trk_ptMax = cms.double(200.),    # maximumum track pT before saturation [GeV]
   	trk_etaMax = cms.double(2.4),    # maximum track eta
	minTrkJetpT=cms.double(-1.),     # min track jet pt to be considered for most energetic zbin finding 
	etaBins=cms.int32(24),
	phiBins=cms.int32(27),
	zBins=cms.int32(1),
	d0_cutNStubs4=cms.double(-1),
	d0_cutNStubs5=cms.double(-1),
	lowpTJetMinTrackMultiplicity=cms.int32(2),#used only for more than 1 z-bins (ie not *prompt*)
        lowpTJetThreshold=cms.double(50.),#used only for more than 1 z-bins (ie not *prompt*)
	highpTJetMinTrackMultiplicity=cms.int32(3),#used only for more than 1 z-bins (ie not *prompt*)
        highpTJetThreshold=cms.double(100.),#used only for more than 1 z-bins (ie not *prompt*)
	displaced=cms.bool(False), #Flag for displaced tracks
        nDisplacedTracks=cms.int32(2)
)

#displaced jets
l1tTrackJetsExtended = l1tTrackJets.clone(
        L1TrackInputTag = cms.InputTag("l1tTrackVertexAssociationProducerExtendedForJets", "Level1TTTracksExtendedSelectedAssociated"),
	minTrkJetpT = 5.,                 # min track jet pt to be considered for most energetic zbin finding
	d0_cutNStubs5 = 0.22,             # -1 excludes nstub>4 from disp tag process
	displaced = True,                  #Flag for displaced tracks
        nDisplacedTracks = 3              #min Ntracks to tag a jet as displaced
)


# selection as presented in the GTT for reference
# d0_cutNStubs4=cms.double(-1),    # -1 excludes nstub=4 from disp tag
# d0_cutNStubs5=cms.double(0.22),  # -1 excludes nstub>4 from disp tag
# lowpTJetMinTrackMultiplicity=cms.int32(2),  #used only on zbin finding
# highpTJetMinTrackMultiplicity=cms.int32(3), #used only on zbin finding
# displaced=cms.bool(True), #Flag for displaced tracks
# nStubs4DisplacedChi2=cms.double(3.3), #Disp tracks selection [trk<cut]
# nStubs4Displacedbend=cms.double(2.3),
# nStubs5DisplacedChi2=cms.double(11.3),
# nStubs5Displacedbend=cms.double(9.8),
# nDisplacedTracks=cms.int32(3) #min Ntracks to tag a jet as displaced
