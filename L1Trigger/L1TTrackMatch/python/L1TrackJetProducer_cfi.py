import FWCore.ParameterSet.Config as cms
from L1Trigger.VertexFinder.VertexProducer_cff import VertexProducer

#prompt jet selection
L1TrackJets = cms.EDProducer('L1TrackJetProducer',
	L1TrackInputTag= cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"),
        L1PVertexCollection = cms.InputTag("VertexProducer", VertexProducer.l1VertexCollectionName.value()),
        MaxDzTrackPV = cms.double( 1.0 ), #max distance from PV;negative=no cut
	trk_zMax = cms.double (15.) ,    # maximum track z
	trk_ptMax = cms.double(200.),    # maximumum track pT before saturation [GeV]
	trk_ptMin = cms.double(3.0),     # minimum track pt [GeV]
   	trk_etaMax = cms.double(2.4),    # maximum track eta
	nStubs4PromptChi2=cms.double(5.0), #Prompt track quality flags for loose/tight
        nStubs4PromptBend=cms.double(1.7),
        nStubs5PromptChi2=cms.double(2.75),
        nStubs5PromptBend=cms.double(3.5),
	trk_nPSStubMin=cms.int32(-1),    # minimum PS stubs, -1 means no cut
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
	nStubs4DisplacedChi2=cms.double(5.0), #Displaced track quality flags for loose/tight
	nStubs4DisplacedBend=cms.double(1.7),
	nStubs5DisplacedChi2=cms.double(2.75),
	nStubs5DisplacedBend=cms.double(3.5),
        nDisplacedTracks=cms.int32(2)
)

#displaced jets
L1TrackJetsExtended = cms.EDProducer('L1TrackJetProducer',
	L1TrackInputTag= cms.InputTag("TTTracksFromExtendedTrackletEmulation", "Level1TTTracks"),
        L1PVertexCollection = cms.InputTag("VertexProducer", VertexProducer.l1VertexCollectionName.value()),
        MaxDzTrackPV = cms.double(5.0),#max track distance from PV;negative=no cut
	trk_zMax = cms.double (15.) ,    # max track z
	trk_ptMax = cms.double(200.),    # maxi track pT before saturation
	trk_ptMin = cms.double(3.0),     # min track pt 
   	trk_etaMax = cms.double(2.4),    # max track eta
        nStubs4PromptChi2=cms.double(5.0), #Prompt track quality flags for loose/tight
        nStubs4PromptBend=cms.double(1.7),
        nStubs5PromptChi2=cms.double(2.75),
        nStubs5PromptBend=cms.double(3.5),
	trk_nPSStubMin=cms.int32(-1),    # min # PS stubs, -1 means no cut
	minTrkJetpT=cms.double(5.),      # min track jet pt to be considered for most energetic zbin finding
	etaBins=cms.int32(24), #number of eta bins
	phiBins=cms.int32(27), #number of phi bins
	zBins=cms.int32(1),    #number of z bins
	d0_cutNStubs4=cms.double(-1),    # -1 excludes nstub=4 from disp tag process
	d0_cutNStubs5=cms.double(0.22),  # -1 excludes nstub>4 from disp tag process
	lowpTJetMinTrackMultiplicity=cms.int32(2),  #relevant only when N of z-bins >1; excludes from the HT calculation jets with low number of very energetic tracks; this cut selects the threshold on number of tracks
        lowpTJetThreshold=cms.double(50.), # this threshold controls the pT of the jet
	highpTJetMinTrackMultiplicity=cms.int32(3), #same as above for a different WP of tracks / pT
        highpTJetThreshold=cms.double(100.),
	displaced=cms.bool(True), #Flag for displaced tracks
	nStubs4DisplacedChi2=cms.double(3.3), #Disp tracks selection [trk<cut]
	nStubs4DisplacedBend=cms.double(2.3),
	nStubs5DisplacedChi2=cms.double(11.3),
	nStubs5DisplacedBend=cms.double(9.8),
        nDisplacedTracks=cms.int32(3) #min Ntracks to tag a jet as displaced
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
