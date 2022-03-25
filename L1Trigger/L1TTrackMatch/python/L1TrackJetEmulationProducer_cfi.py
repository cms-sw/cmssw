import FWCore.ParameterSet.Config as cms
from L1Trigger.VertexFinder.VertexProducer_cff import VertexProducer

L1TrackJetsEmulation = cms.EDProducer('L1TrackJetEmulationProducer',
	L1TrackInputTag= cms.InputTag("L1GTTInputProducer", "Level1TTTracksConverted"),
        VertexInputTag=cms.InputTag("L1VertexFinderEmulator", "l1verticesEmulation"),
	MaxDzTrackPV = cms.double(0.5),
        trk_zMax = cms.double (15.) ,    # maximum track z
	trk_ptMax = cms.double(200.),    # maximumum track pT before saturation [GeV]
	trk_ptMin = cms.double(2.0),     # minimum track pt [GeV]
   	trk_etaMax = cms.double(2.4),    # maximum track eta
	trk_chi2dofMax=cms.double(10.),	 # maximum track chi2/dof
	trk_bendChi2Max=cms.double(2.2), # maximum track bendchi2
	trk_nPSStubMin=cms.int32(-1),    # minimum PS stubs, -1 means no cut
	minTrkJetpT=cms.double(5.),      # minimum track pt to be considered for track jet
	etaBins=cms.int32(24),
	phiBins=cms.int32(27),
	zBins=cms.int32(1),
	d0_cutNStubs4=cms.double(0.15),
	d0_cutNStubs5=cms.double(0.5),
	lowpTJetMinTrackMultiplicity=cms.int32(2),
        lowpTJetMinpT=cms.double(50.),
	highpTJetMinTrackMultiplicity=cms.int32(3),
        highpTJetMinpT=cms.double(100.),
	displaced=cms.bool(False), #Flag for displaced tracks
	nStubs4DisplacedChi2=cms.double(5.0), #Displaced track quality flags for loose/tight
	nStubs4Displacedbend=cms.double(1.7),
	nStubs5DisplacedChi2=cms.double(2.75),
	nStubs5Displacedbend=cms.double(3.5),
	nDisplacedTracks=cms.int32(2) #Number of displaced tracks required per jet
)

L1TrackJetsExtendedEmulation = cms.EDProducer('L1TrackJetEmulationProducer',
	L1TrackInputTag= cms.InputTag("L1GTTInputProducerExtended", "Level1TTTracksExtendedConverted"),
        VertexInputTag=cms.InputTag("L1VertexFinderEmulator", "l1verticesEmulation"),
	MaxDzTrackPV = cms.double(4.0),
        trk_zMax = cms.double (15.) ,    # maximum track z
	trk_ptMax = cms.double(200.),    # maximumum track pT before saturation [GeV]
	trk_ptMin = cms.double(3.0),     # minimum track pt [GeV]
   	trk_etaMax = cms.double(2.4),    # maximum track eta
	trk_chi2dofMax=cms.double(40.),	 # maximum track chi2/dof
	trk_bendChi2Max=cms.double(40.), # maximum track bendchi2
	trk_nPSStubMin=cms.int32(-1),    # minimum # PS stubs, -1 means no cut
	minTrkJetpT=cms.double(5.),      # minimum track pt to be considered for track jet
	etaBins=cms.int32(24),
	phiBins=cms.int32(27),
	zBins=cms.int32(10),
	d0_cutNStubs4=cms.double(-1), # -1 excludes nstub=4 from disp tag
	d0_cutNStubs5=cms.double(0.22),
	lowpTJetMinTrackMultiplicity=cms.int32(2),
        lowpTJetMinpT=cms.double(50.),
	highpTJetMinTrackMultiplicity=cms.int32(3),
        highpTJetMinpT=cms.double(100.),
	displaced=cms.bool(True), #Flag for displaced tracks
	nStubs4DisplacedChi2=cms.double(3.3), #Disp tracks selection [trk<cut]
	nStubs4Displacedbend=cms.double(2.3),
	nStubs5DisplacedChi2=cms.double(11.3),
	nStubs5Displacedbend=cms.double(9.8),
	nDisplacedTracks=cms.int32(3) #min Ntracks to tag a jet as displaced
)
