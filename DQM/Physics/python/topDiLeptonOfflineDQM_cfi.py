import FWCore.ParameterSet.Config as cms

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
      muons = cms.InputTag("muons"),
      elecs = cms.InputTag("gedGsfElectrons"),
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
      #electronId = cms.PSet(
      #  src = cms.InputTag("simpleEleId70cIso"),
      #  #src     = cms.InputTag("eidRobustLoose"),
      #  pattern = cms.int32(1)
      #),
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates
      select = cms.string("pt>10. && abs(eta)<2.4 && abs(gsfTrack.d0)<1. && abs(gsfTrack.dz)<20."),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot
      isolation = cms.string("(dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt<0.2"),
    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras
    muonExtras = cms.PSet(
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates
      select = cms.string("pt>10. && abs(eta)<2.4 && abs(globalTrack.d0)<1. && abs(globalTrack.dz)<20."),
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot
      isolation = cms.string("(isolationR03.sumPt+isolationR03.emEt+isolationR03.hadEt)/pt<0.2"),
    ),
    ## [optional] : when omitted all monitoring plots for jets will
    ## be filled from uncorrected jets
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets
      jetCorrector = cms.string("ak4PFL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak4JetID"),
#        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets
#      select = cms.string("pt>30. & abs(eta)<2.4 & emEnergyFraction>0.01"),
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
      src    = cms.InputTag("muons"),
      select = cms.string("pt>20 & abs(eta)<2.4 & isGlobalMuon & innerTrack.numberOfValidHits>10 & globalTrack.normalizedChi2>-1 & globalTrack.normalizedChi2<10"),
      min    = cms.int32(2),
      max    = cms.int32(2),
    ),
    cms.PSet(
      label  = cms.string("jets/pf:step1"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("ak4PFL2L3"),
#      select = cms.string("pt>30. & abs(eta)<2.4 & emEnergyFraction>0.01"),
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak4JetID"),
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
      muons = cms.InputTag("muons"),
      elecs = cms.InputTag("gedGsfElectrons"),
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
      #electronId = cms.PSet(
      #  src = cms.InputTag("simpleEleId70cIso"),
        #src     = cms.InputTag("eidRobustLoose"),
      #  pattern = cms.int32(1)
      #),
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates
      select = cms.string("pt>500. && abs(eta)<2.4 && abs(gsfTrack.d0)<1. && abs(gsfTrack.dz)<20."),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot
      isolation = cms.string("(dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt<0.2"),
    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras
    muonExtras = cms.PSet(
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates
      select = cms.string("pt>20. && abs(eta)<2.4 && abs(globalTrack.d0)<1. && abs(globalTrack.dz)<20."),
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot
      isolation = cms.string("(isolationR03.sumPt+isolationR03.emEt+isolationR03.hadEt)/pt<0.2"),
    ),
    ## [optional] : when omitted all monitoring plots for jets will
    ## be filled from uncorrected jets
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets
      jetCorrector = cms.string("ak4PFL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak4JetID"),
#        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets
#      select = cms.string("pt>30. & abs(eta)<2.4 & emEnergyFraction>0.01"),
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
      src    = cms.InputTag("muons"),
      select = cms.string("pt>20 & abs(eta)<2.4 & isGlobalMuon & innerTrack.numberOfValidHits>10 & globalTrack.normalizedChi2>-1 & globalTrack.normalizedChi2<10"),
      min    = cms.int32(2),
      max    = cms.int32(2),
    ),
    cms.PSet(
      label  = cms.string("jets/pf:step1"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("ak4PFL2L3"),
#      select = cms.string("pt>30. & abs(eta)<2.4 & emEnergyFraction>0.01"),
      select = cms.string("pt>30. & abs(eta)<2.4 "),
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak4JetID"),
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
      muons = cms.InputTag("muons"),
      elecs = cms.InputTag("gedGsfElectrons"),
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
      #electronId = cms.PSet(
      #  src = cms.InputTag("simpleEleId70cIso"),
      #  #src     = cms.InputTag("eidRobustLoose"),
      #  pattern = cms.int32(1)
      #),
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates
      select = cms.string("pt>10. && abs(eta)<2.4 && abs(gsfTrack.d0)<1. && abs(gsfTrack.dz)<20."),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot
      isolation = cms.string("(dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt<0.2"),
    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras
    muonExtras = cms.PSet(
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates
      select = cms.string("pt>500. && abs(eta)<2.4 && abs(globalTrack.d0)<1. && abs(globalTrack.dz)<20."),
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot
      isolation = cms.string("(isolationR03.sumPt+isolationR03.emEt+isolationR03.hadEt)/pt<0.2"),
    ),
    ## [optional] : when omitted all monitoring plots for jets will
    ## be filled from uncorrected jets
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets
      jetCorrector = cms.string("ak4PFL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak4JetID"),
#        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets
#      select = cms.string("pt>30. & abs(eta)<2.4 & emEnergyFraction>0.01"),
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
      src   = cms.InputTag("gedGsfElectrons"),
#      electronId = cms.PSet(
#        src = cms.InputTag("simpleEleId70cIso"),
#        pattern = cms.int32(1)
#      ),
      select = cms.string("pt>20 & abs(eta)<2.4 & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt<0.17"),
      min = cms.int32(2),
      max = cms.int32(2),
    ),
    cms.PSet(
      label  = cms.string("jets/pf:step1"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("ak4PFL2L3"),
      select = cms.string("pt>30. & abs(eta)<2.4"),
#      select = cms.string("pt>30. & abs(eta)<2.4 & emEnergyFraction>0.01"),
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak4JetID"),
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
      muons = cms.InputTag("muons"),
      elecs = cms.InputTag("gedGsfElectrons"),
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
#      electronId = cms.PSet(
#        src = cms.InputTag("simpleEleId70cIso"),
        #src     = cms.InputTag("eidRobustLoose"),
#        pattern = cms.int32(1)
#      ),
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates
      select = cms.string("pt>10. && abs(eta)<2.4 && abs(gsfTrack.d0)<1. && abs(gsfTrack.dz)<20."),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot
      isolation = cms.string("(dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt<0.2"),
    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras
    muonExtras = cms.PSet(
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates
      select = cms.string("pt>10. && abs(eta)<2.4 && abs(globalTrack.d0)<1. && abs(globalTrack.dz)<20."),
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot
      isolation = cms.string("(isolationR03.sumPt+isolationR03.emEt+isolationR03.hadEt)/pt<0.2"),
    ),
    ## [optional] : when omitted all monitoring plots for jets will
    ## be filled from uncorrected jets
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets
      jetCorrector = cms.string("ak4PFL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak4JetID"),
#        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets
#      select = cms.string("pt>30. & abs(eta)<2.4 & emEnergyFraction>0.01"),
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
      src    = cms.InputTag("muons"),
      select = cms.string("pt>20 & abs(eta)<2.4 & isGlobalMuon & innerTrack.numberOfValidHits>10 & globalTrack.normalizedChi2>-1 & globalTrack.normalizedChi2<10"),
      min    = cms.int32(1),
      max    = cms.int32(1),
    ),
    cms.PSet(
      label = cms.string("elecs:step1"),
      src   = cms.InputTag("gedGsfElectrons"),
      #electronId = cms.PSet(
      #  src = cms.InputTag("simpleEleId70cIso"),
      #  pattern = cms.int32(1)
      #),
      select = cms.string("pt>20 & abs(eta)<2.4 & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt<0.17"),
      min = cms.int32(1),
      max = cms.int32(1),
    ),
    cms.PSet(
      label  = cms.string("jets/pf:step2"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("ak4PFL2L3"),
#      select = cms.string("pt>30. & abs(eta)<2.4 & emEnergyFraction>0.01"),
      select = cms.string("pt>30. & abs(eta)<2.4 "),
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak4JetID"),
#        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      min = cms.int32(2),
      #max = cms.int32(2),
    ),
  ),
)

