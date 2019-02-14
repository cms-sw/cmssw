import FWCore.ParameterSet.Config as cms

TwoLayerJets = cms.EDProducer('TwoLayerJets',
	        L1TrackInputTag= cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),	
		ZMAX = cms.double ( 15. ) ,
		PTMAX = cms.double( 200. ), 
		Etabins=cms.int32(24),
		Phibins=cms.int32(27),
		Zbins=cms.int32(60),
	 	TRK_PTMIN = cms.double(2.0),        # minimum track pt [GeV]
   	 	TRK_ETAMAX = cms.double(2.4)       # maximum track eta
)
