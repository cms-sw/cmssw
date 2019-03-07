import FWCore.ParameterSet.Config as cms

TwoLayerJets = cms.EDProducer('TwoLayerJets',
	        L1TrackInputTag= cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),	
		ZMAX = cms.double ( 15. ) ,
		PTMAX = cms.double( 200. ), # Maximumum track pT before saturation
		Etabins=cms.int32(24),
		Phibins=cms.int32(27),
		Zbins=cms.int32(60),
	 	TRK_PTMIN = cms.double(2.0),        # minimum track pt [GeV]
   	 	TRK_ETAMAX = cms.double(2.4),       # maximum track eta
		CHI2_MAX=cms.double(50.),	
		PromptBendConsistency=cms.double(1.75), #Bend Chi2 Cut for prompt tracks
		D0_Cut=cms.double(0.1),  #Flag for displaced tracks
		nPSStubsMin=cms.int32(2),
		NStubs4Chi2_rz_Loose=cms.double(0.5), ########Displaced track quality flags for loose/tight
		NStubs4Chi2_rphi_Loose=cms.double(0.5),
		NStubs4Displacedbend_Loose=cms.double(1.25),	
		NStubs5Chi2_rz_Loose=cms.double(2.5),
		NStubs5Chi2_rphi_Loose=cms.double(5.0),
		NStubs5Displacedbend_Loose=cms.double(5.0),
		NStubs5Chi2_rz_Tight=cms.double(2.0),
		NStubs5Chi2_rphi_Tight=cms.double(3.5),
		NStubs5Displacedbend_Tight=cms.double(4.0)
)
