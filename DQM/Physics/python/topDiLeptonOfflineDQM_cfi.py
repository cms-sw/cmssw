import FWCore.ParameterSet.Config as cms

looseMuonCut = "muonRef.isNonnull && (muonRef.isGlobalMuon || muonRef.isTrackerMuon) && muonRef.isPFMuon"
looseIsoCut  = "(muonRef.pfIsolationR04.sumChargedHadronPt + max(0., muonRef.pfIsolationR04.sumNeutralHadronEt + muonRef.pfIsolationR04.sumPhotonEt - 0.5 * muonRef.pfIsolationR04.sumPUPt) ) / muonRef.pt < 0.2"
ElelooseIsoCut  = "(gsfElectronRef.pfIsolationVariables.sumChargedHadronPt + max(0., gsfElectronRef.pfIsolationVariables.sumNeutralHadronEt + gsfElectronRef.pfIsolationVariables.sumPhotonEt - 0.5 * gsfElectronRef.pfIsolationVariables.sumPUPt) ) / gsfElectronRef.pt < 0.15"
EletightIsoCut  = "(gsfElectronRef.pfIsolationVariables.sumChargedHadronPt + max(0., gsfElectronRef.pfIsolationVariables.sumNeutralHadronEt + gsfElectronRef.pfIsolationVariables.sumPhotonEt - 0.5 * gsfElectronRef.pfIsolationVariables.sumPUPt) ) / gsfElectronRef.pt < 0.1"


topDiLeptonOfflineDQM = cms.EDAnalyzer("TopDiLeptonOfflineDQM",
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
    directory = cms.string("Physics/Top/TopDiLeptonDQM/"),

    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("pfIsolatedMuonsEI"),
      elecs = cms.InputTag("pfIsolatedElectronsEI"),
      jets  = cms.InputTag("ak4PFJetsCHS"),
      mets  = cms.VInputTag("caloMet", "tcMet", "pfMet")
    ),
    ## [optional] : when omitted the verbosity level is set to STANDARD
    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),
    ## [optional] : when omitted all monitoring plots for electrons
    ## will be filled w/o extras
    elecExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o cut on electronId
      ##electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), cutValue = cms.double(0.0) ),      
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates                                                 
      select = cms.string("pt>20. && abs(eta)<2.5"),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot                                                
      isolation = cms.string(ElelooseIsoCut),
    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras
    muonExtras = cms.PSet(
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates   
      select = cms.string(looseMuonCut + " && muonRef.pt > 10. && abs(muonRef.eta)<2.4"),
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot                                                  
      isolation = cms.string(looseIsoCut),
    ),
    ## [optional] : when omitted all monitoring plots for jets will
    ## be filled from uncorrected jets
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets    
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID                                                   
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak5JetID"),
#        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets
      select = cms.string("pt>30. & abs(eta)<2.4 "), 
    ),
    ## [optional] : when omitted no mass window will be applied
    ## for the same flavor lepton monitoring plots 
    massExtras = cms.PSet(
      lowerEdge = cms.double( 76.0),
      upperEdge = cms.double(106.0)
    ),
    ## [optional] : when omitted all monitoring plots for triggering
    ## will be empty
    #triggerExtras = cms.PSet(
        #src = cms.InputTag("TriggerResults","","HLT"),
        #pathsELECMU = cms.vstring([ 'HLT_Mu9:HLT_Ele15_SW_L1R',
                                    #'HLT_Mu15:HLT_Ele15_SW_L1R',
                                    #'HLT_DoubleMu3:HLT_Ele15_SW_L1R',
                                    #'HLT_Ele15_SW_L1R:HLT_Mu9',
                                    #'HLT_Ele15_SW_L1R:HLT_DoubleMu3']),
        #pathsDIMUON = cms.vstring([ 'HLT_Mu15:HLT_Mu9',
                                    #'HLT_DoubleMu3:HLT_Mu9',
                                    #'HLT_Mu9:HLT_DoubleMu3',
                                    #'HLT_Mu15:HLT_DoubleMu3'])
    #)    
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
    #trigger = cms.PSet(
        #src    = cms.InputTag("TriggerResults","","HLT"),
        #select = cms.vstring(['HLT_Mu9','HLT_Ele15_SW_L1R','HLT_DoubleMu3'])
    #),
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
      label  = cms.string("muons:step0"),
      src    = cms.InputTag("pfIsolatedMuonsEI"),
      select = cms.string(looseMuonCut +" && "+ looseIsoCut + " && muonRef.pt > 20. && abs(muonRef.eta)<2.4"), # CB what to do with iso? CD Added looseIso
      min    = cms.int32(2),
      max    = cms.int32(2),
    ),
    cms.PSet(
      label  = cms.string("jets/pf:step1"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
#      select = cms.string("pt>30. & abs(eta)<2.4 & emEnergyFraction>0.01"),
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak5JetID"),
#        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#     ),
      min = cms.int32(2),
      #max = cms.int32(2),
    )
  )
)



DiMuonDQM = cms.EDAnalyzer("TopDiLeptonOfflineDQM",
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
    directory = cms.string("Physics/Top/TopDiMuonDQM/"),

    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("pfIsolatedMuonsEI"),
      elecs = cms.InputTag("pfIsolatedElectronsEI"),
      jets  = cms.InputTag("ak4PFJetsCHS"),
      mets  = cms.VInputTag("caloMet", "tcMet", "pfMet")
    ),
    ## [optional] : when omitted the verbosity level is set to STANDARD
    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),
    ## [optional] : when omitted all monitoring plots for electrons
    ## will be filled w/o extras
    elecExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o cut on electronId
      ##electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), cutValue = cms.double(0.0) ),      
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates                                                 
      select = cms.string("pt>20. && abs(eta)<2.5"),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot                                                
      isolation = cms.string(ElelooseIsoCut),
    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras
    muonExtras = cms.PSet(
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates   
      select = cms.string(looseMuonCut + " && muonRef.pt > 20. && abs(muonRef.eta)<2.4"),
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot                                                  
      isolation = cms.string(looseIsoCut),
    ),
    ## [optional] : when omitted all monitoring plots for jets will
    ## be filled from uncorrected jets
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets    
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID                                                   
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak5JetID"),
#        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets
      select = cms.string("pt>30. & abs(eta)<2.4 "), 
    ),
    ## [optional] : when omitted no mass window will be applied
    ## for the same flavor lepton monitoring plots 
    massExtras = cms.PSet(
      lowerEdge = cms.double( 76.0),
      upperEdge = cms.double(106.0)
    ),
    ## [optional] : when omitted all monitoring plots for triggering
    ## will be empty
    #triggerExtras = cms.PSet(
        #src = cms.InputTag("TriggerResults","","HLT"),
        #pathsELECMU = cms.vstring([ 'HLT_Mu9:HLT_Ele15_SW_L1R',
                                    #'HLT_Mu15:HLT_Ele15_SW_L1R',
                                    #'HLT_DoubleMu3:HLT_Ele15_SW_L1R',
                                    #'HLT_Ele15_SW_L1R:HLT_Mu9',
                                    #'HLT_Ele15_SW_L1R:HLT_DoubleMu3']),
        #pathsDIMUON = cms.vstring([ 'HLT_Mu15:HLT_Mu9',
                                    #'HLT_DoubleMu3:HLT_Mu9',
                                    #'HLT_Mu9:HLT_DoubleMu3',
                                    #'HLT_Mu15:HLT_DoubleMu3'])
    #)    
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
    #trigger = cms.PSet(
        #src    = cms.InputTag("TriggerResults","","HLT"),
        #select = cms.vstring(['HLT_Mu9','HLT_Ele15_SW_L1R','HLT_DoubleMu3'])
    #),
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
      label  = cms.string("muons:step0"),
      src    = cms.InputTag("pfIsolatedMuonsEI"),
      select = cms.string(looseMuonCut + " && muonRef.pt > 20. && abs(muonRef.eta)<2.4"), # CB what to do with iso?
      min    = cms.int32(2),
      max    = cms.int32(2),
    ),
    cms.PSet(
      label  = cms.string("jets/pf:step1"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30. & abs(eta)<2.4 "), 
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak5JetID"),
#        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      min = cms.int32(2),
      #max = cms.int32(2),
    ),
  ),
)

DiElectronDQM = cms.EDAnalyzer("TopDiLeptonOfflineDQM",
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
    directory = cms.string("Physics/Top/TopDiElectronDQM/"),

    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("pfIsolatedMuonsEI"),
      elecs = cms.InputTag("pfIsolatedElectronsEI"),
      jets  = cms.InputTag("ak4PFJetsCHS"),
      mets  = cms.VInputTag("caloMet", "tcMet", "pfMet")
    ),
    ## [optional] : when omitted the verbosity level is set to STANDARD
    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),
    ## [optional] : when omitted all monitoring plots for electrons
    ## will be filled w/o extras
    elecExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o cut on electronId
      ##electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), cutValue = cms.double(0.0) ),      
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates                                                 
      select = cms.string("pt>20. && abs(eta)<2.5"),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot                                                
      isolation = cms.string(ElelooseIsoCut),
    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras
    muonExtras = cms.PSet(
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates   
      select = cms.string(looseMuonCut + " && muonRef.pt > 20. && abs(muonRef.eta)<2.4"),
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot                                                  
      isolation = cms.string(looseIsoCut),
    ),
    ## [optional] : when omitted all monitoring plots for jets will
    ## be filled from uncorrected jets
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets    
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID                                                   
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak5JetID"),
#        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets
      select = cms.string("pt>30. & abs(eta)<2.4 "), 
    ),
    ## [optional] : when omitted no mass window will be applied
    ## for the same flavor lepton monitoring plots 
    massExtras = cms.PSet(
      lowerEdge = cms.double( 76.0),
      upperEdge = cms.double(106.0)
    ),
    ## [optional] : when omitted all monitoring plots for triggering
    ## will be empty
    #triggerExtras = cms.PSet(
        #src = cms.InputTag("TriggerResults","","HLT"),
        #pathsELECMU = cms.vstring([ 'HLT_Mu9:HLT_Ele15_SW_L1R',
                                    #'HLT_Mu15:HLT_Ele15_SW_L1R',
                                    #'HLT_DoubleMu3:HLT_Ele15_SW_L1R',
                                    #'HLT_Ele15_SW_L1R:HLT_Mu9',
                                    #'HLT_Ele15_SW_L1R:HLT_DoubleMu3']),
        #pathsDIMUON = cms.vstring([ 'HLT_Mu15:HLT_Mu9',
                                    #'HLT_DoubleMu3:HLT_Mu9',
                                    #'HLT_Mu9:HLT_DoubleMu3',
                                    #'HLT_Mu15:HLT_DoubleMu3'])
    #)    
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
    #trigger = cms.PSet(
        #src    = cms.InputTag("TriggerResults","","HLT"),
        #select = cms.vstring(['HLT_Mu9','HLT_Ele15_SW_L1R','HLT_DoubleMu3'])
    #),
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
      label = cms.string("elecs:step0"),
      src   = cms.InputTag("pfIsolatedElectronsEI"),
      ##electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), cutValue = cms.double(0.5) ),      
      select = cms.string("pt>20 & abs(eta)<2.5 && gsfElectronRef.gsfTrack.hitPattern().numberOfHits('MISSING_INNER_HITS') <= 0 && " + ElelooseIsoCut),
      #abs(gsfElectronRef.gsfTrack.d0)<0.04
      min = cms.int32(2),
      max = cms.int32(2),
    ),
    cms.PSet(
      label  = cms.string("jets/pf:step1"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30. & abs(eta)<2.4"), 
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak5JetID"),
#        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      min = cms.int32(2),
      #max = cms.int32(2),
    ),
  ),
)

ElecMuonDQM = cms.EDAnalyzer("TopDiLeptonOfflineDQM",
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
    directory = cms.string("Physics/Top/TopElecMuonDQM/"),

    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("pfIsolatedMuonsEI"),
      elecs = cms.InputTag("pfIsolatedElectronsEI"),
      jets  = cms.InputTag("ak4PFJetsCHS"),
      mets  = cms.VInputTag("caloMet", "tcMet", "pfMet")
    ),
    ## [optional] : when omitted the verbosity level is set to STANDARD
    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),
    ## [optional] : when omitted all monitoring plots for electrons
    ## will be filled w/o extras
    elecExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o cut on electronId
      ##electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), cutValue = cms.double(0.5) ),      
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates                                                 
      select = cms.string("pt>10. && abs(eta)<2.4 && abs(gsfElectronRef.gsfTrack.d0)<1. && abs(gsfElectronRef.gsfTrack.dz)<20."),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot                                                
      isolation = cms.string(ElelooseIsoCut),
    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras
    muonExtras = cms.PSet(
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates
      select = cms.string(looseMuonCut + " && muonRef.pt > 10. && abs(muonRef.eta)<2.4"),
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot                                                  
      isolation = cms.string(looseIsoCut),
    ),
    ## [optional] : when omitted all monitoring plots for jets will
    ## be filled from uncorrected jets
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets    
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID                                                   
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak5JetID"),
#        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets
      select = cms.string("pt>30. & abs(eta)<2.4 "), 
    ),
    ## [optional] : when omitted no mass window will be applied
    ## for the same flavor lepton monitoring plots 
    massExtras = cms.PSet(
      lowerEdge = cms.double( 76.0),
      upperEdge = cms.double(106.0)
    ),
    ## [optional] : when omitted all monitoring plots for triggering
    ## will be empty
    #triggerExtras = cms.PSet(
        #src = cms.InputTag("TriggerResults","","HLT"),
        #pathsELECMU = cms.vstring([ 'HLT_Mu9:HLT_Ele15_SW_L1R',
                                    #'HLT_Mu15:HLT_Ele15_SW_L1R',
                                    #'HLT_DoubleMu3:HLT_Ele15_SW_L1R',
                                    #'HLT_Ele15_SW_L1R:HLT_Mu9',
                                    #'HLT_Ele15_SW_L1R:HLT_DoubleMu3']),
        #pathsDIMUON = cms.vstring([ 'HLT_Mu15:HLT_Mu9',
                                    #'HLT_DoubleMu3:HLT_Mu9',
                                    #'HLT_Mu9:HLT_DoubleMu3',
                                    #'HLT_Mu15:HLT_DoubleMu3'])
    #)    
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
    #trigger = cms.PSet(
        #src    = cms.InputTag("TriggerResults","","HLT"),
        #select = cms.vstring(['HLT_Mu9','HLT_Ele15_SW_L1R','HLT_DoubleMu3'])
    #),
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
      label  = cms.string("muons:step0"),
      src    = cms.InputTag("pfIsolatedMuonsEI"),
      select = cms.string(looseMuonCut + " && " + looseIsoCut + " && muonRef.pt > 20. && abs(muonRef.eta)<2.4"), # CB what to do with iso? CD Added looseIsoCut
      min    = cms.int32(1),
      max    = cms.int32(1),
    ),
    cms.PSet(
      label = cms.string("elecs:step1"),
      src   = cms.InputTag("pfIsolatedElectronsEI"),
      ##electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), cutValue = cms.double(0.5) ),      
      select = cms.string("pt>20 & abs(eta)<2.5 && "+ElelooseIsoCut),
      min = cms.int32(1),
      max = cms.int32(1),
    ),
    cms.PSet(
      label  = cms.string("jets/pf:step2"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30. & abs(eta)<2.4 "), 
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak5JetID"),
#        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      min = cms.int32(2),
      #max = cms.int32(2),
    ),
  ),
)

