import FWCore.ParameterSet.Config as cms

ecalPresampleTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        doPulseMaxCheck = cms.untracked.bool(True),
        pulseMaxPosition = cms.untracked.int32(5),
        nSamples = cms.untracked.int32(3)
    ),
    MEs = cms.untracked.PSet(
        Pedestal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineTask/Gain12/%(prefix)sPOT pedestal %(sm)s G12'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('2D distribution of mean presample value.')
        ),
        PedestalByLS = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineTask/Gain12/%(prefix)sPOT pedestal by LS %(sm)s G12'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('2D distribution of mean presample value for "current" LS.')
        ),
  	PedestalProjEtaG1 = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/%(prefix)sOT Pedestal RMS values from DB %(suffix)s eta projection Gain1'),
            kind = cms.untracked.string('TProfile'),
	    yaxis = cms.untracked.PSet(
	    	title = cms.untracked.string('Pedestal RMS') 
	    ), 
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjEta'),
            description = cms.untracked.string('Projection of Pedestal rms values from DB')
       ),
       PedestalProjEtaG6 = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/%(prefix)sOT Pedestal RMS values from DB %(suffix)s eta projection Gain6'),
            kind = cms.untracked.string('TProfile'),
	    yaxis = cms.untracked.PSet(
                title = cms.untracked.string('Pedestal RMS') 
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjEta'),
            description = cms.untracked.string('Projection of Pedestal rms values from DB')
       ),

      PedestalProjEtaG12 = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/%(prefix)sOT Pedestal RMS values from DB %(suffix)s eta projection Gain12'),
            kind = cms.untracked.string('TProfile'),
	    yaxis = cms.untracked.PSet(
                title = cms.untracked.string('Pedestal RMS') 
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjEta'),
            description = cms.untracked.string('Projection of Pedestal rms values from DB')
       )

  )
)
