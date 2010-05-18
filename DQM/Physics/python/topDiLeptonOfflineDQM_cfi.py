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
      elecs = cms.InputTag("gsfElectrons"),
      jets  = cms.InputTag("ak5CaloJets"),
      mets  = cms.VInputTag("met", "tcMet", "pfMet")
    ),
    ## [optional] : when omitted the verbosity level is set to STANDARD
    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),
    ## [optional] : when omitted all monitoring plots for the electron
    ## will be filled w/o preselection
    elecExtras = cms.PSet(
      select = cms.string("pt>5 && abs(eta)<2.4 && abs(gsfTrack.d0)<1 && abs(gsfTrack.dz)<20"),
      isolation = cms.string("(dr03TkSumPt+dr04EcalRecHitSumEt+dr04HcalTowerSumEt)/pt<0.2"),
      electronId = cms.InputTag("eidRobustTight") ## used eidLoose
    ),
    ## [optional] : when omitted all monitoring plots for the muon
    ## will be filled w/o preselection
    muonExtras = cms.PSet(
      select = cms.string("pt>1 && abs(eta)<2.4 && abs(globalTrack.d0)<1 && abs(globalTrack.dz)<20"),
      isolation = cms.string("(isolationR03.sumPt+isolationR03.emEt+isolationR03.hadEt)/pt<0.2"),
    ),
    ## [optional] : when omitted no mass window will be applied
    ## for the same flavor lepton monitoring plots 
    massExtras = cms.PSet(
      lowerEdge = cms.double(3.0),
      upperEdge = cms.double(3.2)
    ),
    ## [optional] : when omitted all monitoring plots for jets will
    ## be filled from uncorrected jets
    jetExtras = cms.PSet(
      jetCorrector = cms.string("ak5CaloL2L3")
    ),
    ## [optional] : when omitted all monitoring plots for triggering
    ## will be empty
    triggerExtras = cms.PSet(
      src = cms.InputTag("TriggerResults","","HLT"),
      pathsELECMU = cms.vstring(['HLT_Mu9:HLT_Ele15_SW_L1R',
                                 'HLT_Mu15:HLT_Ele15_SW_L1R',
                                 'HLT_DoubleMu3:HLT_Ele15_SW_L1R',
                                 'HLT_Ele15_SW_L1R:HLT_Mu9',
                                 'HLT_Ele15_SW_L1R:HLT_DoubleMu3']),
      pathsDIMUON = cms.vstring(['HLT_Mu15:HLT_Mu9',
                                 'HLT_DoubleMu3:HLT_Mu9',
                                 'HLT_Mu9:HLT_DoubleMu3',
                                 'HLT_Mu15:HLT_DoubleMu3'])
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
      select = cms.vstring(['HLT_Mu9','HLT_Ele15_SW_L1R','HLT_DoubleMu3'])
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
    cms.PSet(
      ## [mandatory] : 'jets' defines the objects to
      ## select on, 'step0' labels the histograms;
      ## instead of 'step0' you can choose any label
      label  = cms.string("empty:step0")
    ),
  ),
)
