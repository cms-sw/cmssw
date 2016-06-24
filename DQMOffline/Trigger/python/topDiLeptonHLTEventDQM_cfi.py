import FWCore.ParameterSet.Config as cms

topDiLeptonHLTOfflineDQM = cms.EDAnalyzer("TopDiLeptonHLTOfflineDQM",
  ## ------------------------------------------------------
  ## SETUP
  ##
  ## configuration of the MonitoringEnsemble(s)
  ## [mandatory] : optional PSets may be omitted
  ##
  setup = cms.PSet(
    ## sub-directory to write the monitor histograms to
    ## [mandatory] : should not be changed w/o explicit 
    ## communication to TopCom!
    directory = cms.string("HLT/TopHLTOffline/Top/DiLeptonic/"),

    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("muons"),
      elecs = cms.InputTag("gedGsfElectrons"),
      jets  = cms.InputTag("ak4PFJetsCHS"),
      mets  = cms.VInputTag("met", "tcMet", "pfMet")
    ),
    ## [optional] : when omitted all monitoring plots for electrons
    ## will be filled w/o extras
    elecExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o cut on electronId
      #electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), pattern = cms.int32(1) ),
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates                                                 
      select = cms.string("pt>20 & abs(eta)<2.5"),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot                                                
      isolation = cms.string("(dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt<0.15"),
    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras
    muonExtras = cms.PSet(
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates   
      select = cms.string("pt>20 & abs(eta)<2.4 & isPFMuon & (isGlobalMuon || isTrackerMuon)"),
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot                                                  
      isolation = cms.string("(pfIsolationR04.sumChargedHadronPt+pfIsolationR04.sumPhotonEt+pfIsolationR04.sumNeutralHadronEt)/pt<0.2"),
    ),
    ## [optional] : when omitted all monitoring plots for jets will
    ## be filled from uncorrected jets
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets    
      #jetCorrector = cms.string("ak4PFCHSL2L3"),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets
      select = cms.string("pt>30. & abs(eta)<2.5"), 
    ),
    ## [optional] : when omitted no mass window will be applied
    ## for the same flavor lepton monitoring plots 
    massExtras = cms.PSet(
      lowerEdge = cms.double( 70.0),
      upperEdge = cms.double(110.0)
    ),
    ## [optional] : when omitted all monitoring plots for triggering
    ## will be empty
    triggerExtras = cms.PSet(
		src = cms.InputTag("TriggerResults","","HLT"),
### Updating to HLT paths to be monitored by TOP PAG in 2016
		pathsELECMU = cms.vstring([ 'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v',
                                    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v',
                                    'HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v',
                                    'HLT_Mu8_TrkIsoVVL_Ele17_CaloIdL_TrackIdL_IsoVL_v']),
		pathsDIMUON = cms.vstring([ 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v',
                                    'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v']),
		pathsDIELEC = cms.vstring([ 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v',
                                    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v'])
	)																	  
  ),
                                  
  ## ------------------------------------------------------
  ## PRESELECTION
  ##
  ## setup of the event preselection, which will not
  ## be monitored
  ## [mandatory] : but may be empty
  ##
  preselection = cms.PSet(
    ## [optional] : when omitted no preselection is applied
    trigger = cms.PSet(
        src    = cms.InputTag("TriggerResults","","HLT"),
### Updating to HLT paths to be monitored by TOP PAG in 2016                                                                                                                 
		select = cms.vstring(['HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v', 'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v', 'HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v', 'HLT_Mu8_TrkIsoVVL_Ele17_CaloIdL_TrackIdL_IsoVL_v''HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v', 'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v', 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v', 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v'])
    ),
    ## [optional] : when omitted no preselection is applied
    vertex = cms.PSet(
      src    = cms.InputTag("offlinePrimaryVertices"),
      select = cms.string('abs(x)<1. && abs(y)<1. && abs(z)<20. && tracksSize>3 && !isFake')
    )
  ),
  
  ## ------------------------------------------------------    
  ## SELECTION
  ##
  ## monitor histrograms are filled after each selection
  ## step, the selection is applied in the order defined
  ## by this vector
  ## [mandatory] : may be empty or contain an arbitrary
  ## number of PSets as given below:
  ##
  selection = cms.VPSet(
    #cms.PSet(
      ### [mandatory] : 'jets' defines the objects to
      ### select on, 'step0' labels the histograms;
      ### instead of 'step0' you can choose any label
      #label  = cms.string("empty:step0")
    #),
    cms.PSet(
      label  = cms.string("Hlt:step0"),
      src    = cms.InputTag(""),
      select = cms.string(""),
      min    = cms.int32(0),
      max    = cms.int32(0),
    ),
    cms.PSet(
      label  = cms.string("jets/pf:step1"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      #jetCorrector = cms.string("ak4PFCHSL2L3"),
      select = cms.string("pt>30. & abs(eta)<2.5"),
      min = cms.int32(2),
    )
  )
)



DiMuonHLTOfflineDQM = cms.EDAnalyzer("TopDiLeptonHLTOfflineDQM",
  ## ------------------------------------------------------
  ## SETUP
  ##
  ## configuration of the MonitoringEnsemble(s)
  ## [mandatory] : optional PSets may be omitted
  ##
  setup = cms.PSet(
    ## sub-directory to write the monitor histograms to
    ## [mandatory] : should not be changed w/o explicit 
    ## communication to TopCom!
    directory = cms.string("HLT/TopHLTOffline/Top/DiMuon/"),

    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("muons"),
      elecs = cms.InputTag("gedGsfElectrons"),
      jets  = cms.InputTag("ak4PFJetsCHS"),
      mets  = cms.VInputTag("met", "tcMet", "pfMet")
    ),
    ## [optional] : when omitted all monitoring plots for electrons
    ## will be filled w/o extras
    elecExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o cut on electronId
      #electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), pattern = cms.int32(1) ),
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates                                                 
      select = cms.string("pt>20 & abs(eta)<2.5"),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot                                                
      isolation = cms.string("(dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt<0.15"),
    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras
    muonExtras = cms.PSet(
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates   
      select = cms.string("pt>20 & abs(eta)<2.4 & isPFMuon & (isGlobalMuon || isTrackerMuon)"),
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot                                                  
      isolation = cms.string("(pfIsolationR04.sumChargedHadronPt+pfIsolationR04.sumPhotonEt+pfIsolationR04.sumNeutralHadronEt)/pt<0.2"),
    ),
    ## [optional] : when omitted all monitoring plots for jets will
    ## be filled from uncorrected jets
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets    
      #jetCorrector = cms.string("ak4PFCHSL2L3"),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets
      select = cms.string("pt>30. & abs(eta)<2.5"), 
    ),
    ## [optional] : when omitted no mass window will be applied
    ## for the same flavor lepton monitoring plots 
    massExtras = cms.PSet(
      lowerEdge = cms.double( 70.0),
      upperEdge = cms.double(110.0)
    ),
    ## [optional] : when omitted all monitoring plots for triggering
    ## will be empty
    triggerExtras = cms.PSet(
        src = cms.InputTag("TriggerResults","","HLT"),
### Updating to HLT paths to be monitored by TOP PAG in 2016                                                                                                                 
		pathsELECMU = cms.vstring([ 'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v',
                                    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v',
                                    'HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v', 
                                    'HLT_Mu8_TrkIsoVVL_Ele17_CaloIdL_TrackIdL_IsoVL_v']),
		pathsDIMUON = cms.vstring([ 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v',
                                    'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_Dz_v']),
		pathsDIELEC = cms.vstring([ 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v', 
                                    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v'])
    )    
  ),                                
  ## ------------------------------------------------------
  ## PRESELECTION
  ##
  ## setup of the event preselection, which will not
  ## be monitored
  ## [mandatory] : but may be empty
  ##
  preselection = cms.PSet(
    ## [optional] : when omitted no preselection is applied
    trigger = cms.PSet(
        src    = cms.InputTag("TriggerResults","","HLT"),
        select = cms.vstring(['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v',
                              'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v']),
    ),
    ## [optional] : when omitted no preselection is applied
    vertex = cms.PSet(
      src    = cms.InputTag("offlinePrimaryVertices"),
      select = cms.string('abs(x)<1. && abs(y)<1. && abs(z)<20. && tracksSize>3 && !isFake')
    )
  ),
  
  ## ------------------------------------------------------    
  ## SELECTION
  ##
  ## monitor histrograms are filled after each selection
  ## step, the selection is applied in the order defined
  ## by this vector
  ## [mandatory] : may be empty or contain an arbitrary
  ## number of PSets as given below:
  ##
  selection = cms.VPSet(
    #cms.PSet(
      ### [mandatory] : 'jets' defines the objects to
      ### select on, 'step0' labels the histograms;
      ### instead of 'step0' you can choose any label
      #label  = cms.string("empty:step0")
    #),
    cms.PSet(
      label  = cms.string("Hlt:step0"),
      src    = cms.InputTag(""),
      select = cms.string(""),
      min    = cms.int32(0),
      max    = cms.int32(0),
    ),

    cms.PSet(
      label  = cms.string("muons:step1"),
      src    = cms.InputTag("muons"),
      select = cms.string("pt>20 & abs(eta)<2.4 & isPFMuon & (isGlobalMuon || isTrackerMuon) & (isolationR03.sumPt+isolationR03.emEt+isolationR03.hadEt)/pt<0.2"),
      min    = cms.int32(2),
      max    = cms.int32(2),
    ),
    cms.PSet(
      label  = cms.string("jets/pf:step2"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      #jetCorrector = cms.string("ak4PFCHSL2L3"),
      select = cms.string("pt>30. & abs(eta)<2.5"),
      min = cms.int32(2),
    ),
  ),
)

DiElectronHLTOfflineDQM = cms.EDAnalyzer("TopDiLeptonHLTOfflineDQM",
  ## ------------------------------------------------------
  ## SETUP
  ##
  ## configuration of the MonitoringEnsemble(s)
  ## [mandatory] : optional PSets may be omitted
  ##
  setup = cms.PSet(
    ## sub-directory to write the monitor histograms to
    ## [mandatory] : should not be changed w/o explicit 
    ## communication to TopCom!
    directory = cms.string("HLT/TopHLTOffline/Top/DiElectron/"),

    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("muons"),
      elecs = cms.InputTag("gedGsfElectrons"),
      jets  = cms.InputTag("ak4PFJetsCHS"),
      mets  = cms.VInputTag("met", "tcMet", "pfMet")
    ),
    ## [optional] : when omitted all monitoring plots for electrons
    ## will be filled w/o extras
    elecExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o cut on electronId
      #electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), pattern = cms.int32(1) ),
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates                                                 
      select = cms.string("pt>20 & abs(eta)<2.5"),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot                                                
      isolation = cms.string("(dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt<0.15"),
    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras
    muonExtras = cms.PSet(
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates   
      select = cms.string("pt>20 & abs(eta)<2.4 & isPFMuon & (isGlobalMuon || isTrackerMuon)"),
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot                                                  
      isolation = cms.string("(pfIsolationR04.sumChargedHadronPt+pfIsolationR04.sumPhotonEt+pfIsolationR04.sumNeutralHadronEt)/pt<0.2"),
    ),
    ## [optional] : when omitted all monitoring plots for jets will
    ## be filled from uncorrected jets
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets    
      #jetCorrector = cms.string("ak4PFCHSL2L3"),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets
      select = cms.string("pt>30. & abs(eta)<2.5"), 
    ),
    ## [optional] : when omitted no mass window will be applied
    ## for the same flavor lepton monitoring plots 
    massExtras = cms.PSet(
      lowerEdge = cms.double( 70.0),
      upperEdge = cms.double(110.0)
    ),
    ## [optional] : when omitted all monitoring plots for triggering
    ## will be empty
    triggerExtras = cms.PSet(
        src = cms.InputTag("TriggerResults","","HLT"),

### Updating to HLT paths to be monitored by TOP PAG in 2016                                                                                                
		pathsELECMU = cms.vstring([ 'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v',
                                    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v',
                                    'HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v', 
                                    'HLT_Mu8_TrkIsoVVL_Ele17_CaloIdL_TrackIdL_IsoVL_v']),
		pathsDIMUON = cms.vstring([ 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v',
                                    'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v']),
		pathsDIELEC = cms.vstring([ 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v',
                                    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v'])
	)	
  ),
                                  
  ## ------------------------------------------------------
  ## PRESELECTION
  ##
  ## setup of the event preselection, which will not
  ## be monitored
  ## [mandatory] : but may be empty
  ##
  preselection = cms.PSet(
    ## [optional] : when omitted no preselection is applied
    trigger = cms.PSet(
        src    = cms.InputTag("TriggerResults","","HLT"),
        select = cms.vstring(['HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v', 
                              'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v'])
    ),
    ## [optional] : when omitted no preselection is applied
    vertex = cms.PSet(
      src    = cms.InputTag("offlinePrimaryVertices"),
      select = cms.string('abs(x)<1. && abs(y)<1. && abs(z)<20. && tracksSize>3 && !isFake')
    )
  ),
  
  ## ------------------------------------------------------    
  ## SELECTION
  ##
  ## monitor histrograms are filled after each selection
  ## step, the selection is applied in the order defined
  ## by this vector
  ## [mandatory] : may be empty or contain an arbitrary
  ## number of PSets as given below:
  ##
  selection = cms.VPSet(
    #cms.PSet(
      ### [mandatory] : 'jets' defines the objects to
      ### select on, 'step0' labels the histograms;
      ### instead of 'step0' you can choose any label
      #label  = cms.string("empty:step0")
    #),
    cms.PSet(
      label  = cms.string("Hlt:step0"),
      src    = cms.InputTag(""),
      select = cms.string(""),
      min    = cms.int32(0),
      max    = cms.int32(0),
    ),

    cms.PSet(
      label = cms.string("elecs:step1"),
      src   = cms.InputTag("gedGsfElectrons"),
      #electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), pattern = cms.int32(1) ),
      select = cms.string("pt>20 & abs(eta)<2.5 & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt<0.15"),
      min = cms.int32(2),
      max = cms.int32(2),
    ),
    cms.PSet(
      label  = cms.string("jets/pf:step2"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      #jetCorrector = cms.string("ak4PFCHSL2L3"),
      select = cms.string("pt>30. & abs(eta)<2.5"),
      min = cms.int32(2),
    ),
  ),
)

ElecMuonHLTOfflineDQM = cms.EDAnalyzer("TopDiLeptonHLTOfflineDQM",
  ## ------------------------------------------------------
  ## SETUP
  ##
  ## configuration of the MonitoringEnsemble(s)
  ## [mandatory] : optional PSets may be omitted
  ##
  setup = cms.PSet(
    ## sub-directory to write the monitor histograms to
    ## [mandatory] : should not be changed w/o explicit 
    ## communication to TopCom!
    directory = cms.string("HLT/TopHLTOffline/Top/ElecMuon/"),

    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("muons"),
      elecs = cms.InputTag("gedGsfElectrons"),
      jets  = cms.InputTag("ak4PFJetsCHS"),
      mets  = cms.VInputTag("met", "tcMet", "pfMet")
    ),
    ## [optional] : when omitted all monitoring plots for electrons
    ## will be filled w/o extras
    elecExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o cut on electronId
      #electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), pattern = cms.int32(1) ),
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates                                                 
      select = cms.string("pt>20 & abs(eta)<2.5"),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot                                                
      isolation = cms.string("(dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt<0.15"),
    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras
    muonExtras = cms.PSet(
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates   
      select = cms.string("pt>20 & abs(eta)<2.4 & isPFMuon & (isGlobalMuon || isTrackerMuon)"),
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot                                                  
      isolation = cms.string("(pfIsolationR04.sumChargedHadronPt+pfIsolationR04.sumPhotonEt+pfIsolationR04.sumNeutralHadronEt)/pt<0.2"),
    ),
    ## [optional] : when omitted all monitoring plots for jets will
    ## be filled from uncorrected jets
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets    
      #jetCorrector = cms.string("ak4PFCHSL2L3"),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets
      select = cms.string("pt>30. & abs(eta)<2.5"), 
    ),
    ## [optional] : when omitted no mass window will be applied
    ## for the same flavor lepton monitoring plots 
    massExtras = cms.PSet(
      lowerEdge = cms.double( 70.0),
      upperEdge = cms.double(110.0)
    ),
    ## [optional] : when omitted all monitoring plots for triggering
    ## will be empty
    triggerExtras = cms.PSet(
        src = cms.InputTag("TriggerResults","","HLT"),
### Updating to HLT paths to be monitored by TOP PAG in 2016                                                                                                                 
		pathsELECMU = cms.vstring([ 'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v',
                                    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v',
                                    'HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v', 
                                    'HLT_Mu8_TrkIsoVVL_Ele17_CaloIdL_TrackIdL_IsoVL_v']),
		pathsDIMUON = cms.vstring([ 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v',
                                    'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v']),
		pathsDIELEC = cms.vstring([ 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v',
                                    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v'])
	)    
  ),
                                  
  ## ------------------------------------------------------
  ## PRESELECTION
  ##
  ## setup of the event preselection, which will not
  ## be monitored
  ## [mandatory] : but may be empty
  ##
  preselection = cms.PSet(
    ## [optional] : when omitted no preselection is applied
    trigger = cms.PSet(
        src    = cms.InputTag("TriggerResults","","HLT"),
### Updating to HLT paths to be monitored by TOP PAG in 2016 
		select = cms.vstring(['HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v',                                                                                            
                              'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v',
                              'HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v',
                              'HLT_Mu8_TrkIsoVVL_Ele17_CaloIdL_TrackIdL_IsoVL_v'])
	),
    ## [optional] : when omitted no preselection is applied
    vertex = cms.PSet(
      src    = cms.InputTag("offlinePrimaryVertices"),
      select = cms.string('abs(x)<1. && abs(y)<1. && abs(z)<20. && tracksSize>3 && !isFake')
    )
  ),
  
  ## ------------------------------------------------------    
  ## SELECTION
  ##
  ## monitor histrograms are filled after each selection
  ## step, the selection is applied in the order defined
  ## by this vector
  ## [mandatory] : may be empty or contain an arbitrary
  ## number of PSets as given below:
  ##
  selection = cms.VPSet(
    #cms.PSet(
      ### [mandatory] : 'jets' defines the objects to
      ### select on, 'step0' labels the histograms;
      ### instead of 'step0' you can choose any label
      #label  = cms.string("empty:step0")
    #),
    cms.PSet(
      label  = cms.string("Hlt:step0"),
      src    = cms.InputTag(""),
      select = cms.string(""),
      min    = cms.int32(0),
      max    = cms.int32(0),
    ),

    cms.PSet(
      label  = cms.string("muons:step1"),
      src    = cms.InputTag("muons"),
      select = cms.string("pt>20 & abs(eta)<2.4 & isPFMuon & (isGlobalMuon || isTrackerMuon) & (pfIsolationR04.sumChargedHadronPt+pfIsolationR04.sumPhotonEt+pfIsolationR04.sumNeutralHadronEt)/pt<0.2"),
      min    = cms.int32(1),
      max    = cms.int32(1),
    ),
    cms.PSet(
      label = cms.string("elecs:step2"),
      src   = cms.InputTag("gedGsfElectrons"),
      #electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), pattern = cms.int32(1) ),
      select = cms.string("pt>20 & abs(eta)<2.5 & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt<0.15"),
      min = cms.int32(1),
      max = cms.int32(1),
    ),
    cms.PSet(
      label  = cms.string("jets/pf:step3"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      #jetCorrector = cms.string("ak4PFCHSL2L3"),
      select = cms.string("pt>30. & abs(eta)<2.5"),
      min = cms.int32(2),
    ),
  ),
)
