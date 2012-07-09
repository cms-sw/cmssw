import FWCore.ParameterSet.Config as cms

SingleTopSingleMuonTriggerDQM = cms.EDAnalyzer("TopSingleLeptonDQM",
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
    directory = cms.string("Physics/SingleTop/SingleTopTriggerDQM/"),
    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("muons"),
      elecs = cms.InputTag("gsfElectrons"),
      jets  = cms.InputTag("ak5CaloJets"),
      mets  = cms.VInputTag("met", "tcMet", "pfMet"),
      pvs   = cms.InputTag("offlinePrimaryVertices")

    ),
    ## [optional] : when omitted the verbosity level is set to STANDARD
    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),
    ## [optional] : when omitted all monitoring plots for primary vertices
    ## will be filled w/o extras
    pvExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the primary vertex candidates                                                                                            
      select = cms.string("abs(x)<1. & abs(y)<1. & abs(z)<20. & tracksSize>3 & !isFake")
    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras                                           
    muonExtras = cms.PSet(
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates                                                
      select    = cms.string("pt>20 & abs(eta)<2.1 & isGlobalMuon & innerTrack.numberOfValidHits>10 & globalTrack.normalizedChi2>-1 & globalTrack.normalizedChi2<10 & (isolationR03.sumPt+isolationR03.emEt+isolationR03.hadEt)/pt<0.1"),  
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot                                                    
      isolation = cms.string("(isolationR03.sumPt+isolationR03.emEt+isolationR03.hadEt)/pt<0.1")
    ),
    ## [optional] : when omitted all monitoring plots for jets
    ## will be filled w/o extras
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets
      jetCorrector = cms.string("ak5CaloL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID                                                                                                   
      jetID  = cms.PSet(
        label  = cms.InputTag("ak5JetID"),
        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
      ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets                                                
      select = cms.string("pt>30 & abs(eta)<4.5& emEnergyFraction>0.01"),
      ## when omitted monitor histograms for b-tagging will not be filled                                                                                                   
      jetBTaggers  = cms.PSet(
        trackCountingEff = cms.PSet(
          label = cms.InputTag("trackCountingHighEffBJetTags" ),
          workingPoint = cms.double(1.25)
        ),
        trackCountingPur = cms.PSet(
          label = cms.InputTag("trackCountingHighPurBJetTags" ),
          workingPoint = cms.double(3.00)
        ),
        secondaryVertex  = cms.PSet(
          label = cms.InputTag("simpleSecondaryVertexHighEffBJetTags"),
          workingPoint = cms.double(2.05)
        )
      ),                                                
    ),
    ## [optional] : when omitted no mass window will be applied
    ## for the W mass before filling the event monitoring plots
    massExtras = cms.PSet(
      lowerEdge = cms.double( 70.),
      upperEdge = cms.double(110.)
    ),
    ## [optional] : when omitted the monitoring plots for triggering
    ## will be empty
    triggerExtras = cms.PSet(
      src   = cms.InputTag("TriggerResults","","HLT"),
     paths = cms.vstring(['HLT_Mu3:HLT_QuadJet15U',
                          'HLT_Mu5:HLT_QuadJet15U',
                          'HLT_Mu7:HLT_QuadJet15U',
                          'HLT_Mu9:HLT_QuadJet15U',
                          'HLT_Mu11:HLT_QuadJet15U'])      
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
   trigger = cms.PSet(
      src    = cms.InputTag("TriggerResults","","HLT"),
      select = cms.vstring(['HLT_Iso10Mu20_eta2p1_CentralPFJet30_BTagIPIter_v1'])#ONLY ONE PATH
    ),
    ## [optional] : when omitted no preselection is applied
    vertex = cms.PSet(
      src    = cms.InputTag("offlinePrimaryVertices"),
      select = cms.string('abs(x)<1. & abs(y)<1. & abs(z)<20. & tracksSize>3 & !isFake')
    )                                       
  ), 
  ## ------------------------------------------------------
  ## SELECTION
  ##
  ## monitor histrograms are filled after each selection
  ## step, the selection is applied in the order defined
  ## by this vector
  ## [mandatory] : may be empty or contain an arbitrary
  ## number of PSets
  selection = cms.VPSet(
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
      select = cms.string("pt>20 & abs(eta)<2.1 & isGlobalMuon & innerTrack.numberOfValidHits>10 & globalTrack.normalizedChi2>-1 & globalTrack.normalizedChi2<10 & (isolationR03.sumPt+isolationR03.emEt+isolationR03.hadEt)/pt<0.1"),       
      min    = cms.int32(1),
      max    = cms.int32(1),
    ),
    cms.PSet(
      label  = cms.string("jets/calo:step2"),
      src    = cms.InputTag("ak5CaloJets"),
      jetCorrector = cms.string("ak5CaloL2L3"),
      select = cms.string("pt>30 & abs(eta)<4.5 & emEnergyFraction>0.01"),
      jetID  = cms.PSet(
        label  = cms.InputTag("ak5JetID"),
        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
      ),
      min = cms.int32(1),
    ), 
    cms.PSet(
      label  = cms.string("jets/calo:step3"),
      src    = cms.InputTag("ak5CaloJets"),
      jetCorrector = cms.string("ak5CaloL2L3"),
      select = cms.string("pt>30 & abs(eta)<4.5 & emEnergyFraction>0.01"),
      jetID  = cms.PSet(
        label  = cms.InputTag("ak5JetID"),
        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
      ),
      min = cms.int32(2),
    ), 
  )
)

SingleTopSingleElectronTriggerDQM = cms.EDAnalyzer("TopSingleLeptonDQM",
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
    directory = cms.string("Physics/SingleTop/SingleTopTriggerDQM/"),
    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("muons"),
      elecs = cms.InputTag("gsfElectrons"),
      jets  = cms.InputTag("ak5CaloJets"),
      mets  = cms.VInputTag("met", "tcMet", "pfMet"),
      pvs   = cms.InputTag("offlinePrimaryVertices")

    ),
    ## [optional] : when omitted the verbosity level is set to STANDARD
    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),
    ## [optional] : when omitted all monitoring plots for primary vertices
    ## will be filled w/o extras
    pvExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the primary vertex candidates                                                                                            
      select = cms.string("abs(x)<1. & abs(y)<1. & abs(z)<20. & tracksSize>3 & !isFake")
    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras                                           
    muonExtras = cms.PSet(
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates                                                
      select    = cms.string("pt>20 & abs(eta)<2.1 & isGlobalMuon & innerTrack.numberOfValidHits>10 & globalTrack.normalizedChi2>-1 & globalTrack.normalizedChi2<10 & (isolationR03.sumPt+isolationR03.emEt+isolationR03.hadEt)/pt<0.1"),  
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot                                                    
      isolation = cms.string("(isolationR03.sumPt+isolationR03.emEt+isolationR03.hadEt)/pt<0.1")
    ),
    ## [optional] : when omitted all monitoring plots for jets
    ## will be filled w/o extras
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets
      jetCorrector = cms.string("ak5CaloL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID                                                                                                   
      jetID  = cms.PSet(
        label  = cms.InputTag("ak5JetID"),
        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
      ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets                                                
      select = cms.string("pt>30 & abs(eta)<4.5& emEnergyFraction>0.01"),
      ## when omitted monitor histograms for b-tagging will not be filled                                                                                                   
      jetBTaggers  = cms.PSet(
        trackCountingEff = cms.PSet(
          label = cms.InputTag("trackCountingHighEffBJetTags" ),
          workingPoint = cms.double(1.25)
        ),
        trackCountingPur = cms.PSet(
          label = cms.InputTag("trackCountingHighPurBJetTags" ),
          workingPoint = cms.double(3.00)
        ),
        secondaryVertex  = cms.PSet(
          label = cms.InputTag("simpleSecondaryVertexHighEffBJetTags"),
          workingPoint = cms.double(2.05)
        )
      ),                                                
    ),
    ## [optional] : when omitted no mass window will be applied
    ## for the W mass before filling the event monitoring plots
    massExtras = cms.PSet(
      lowerEdge = cms.double( 70.),
      upperEdge = cms.double(110.)
    ),
    ## [optional] : when omitted the monitoring plots for triggering
    ## will be empty
    triggerExtras = cms.PSet(
      src   = cms.InputTag("TriggerResults","","HLT"),
     paths = cms.vstring(['HLT_Mu3:HLT_QuadJet15U',
                          'HLT_Mu5:HLT_QuadJet15U',
                          'HLT_Mu7:HLT_QuadJet15U',
                          'HLT_Mu9:HLT_QuadJet15U',
                          'HLT_Mu11:HLT_QuadJet15U'])      
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
   trigger = cms.PSet(
      src    = cms.InputTag("TriggerResults","","HLT"),
      select = cms.vstring(['HLT_Iso10Mu20_eta2p1_CentralPFJet30_BTagIPIter_v1'])#ONLY ONE PATH
    ),
    ## [optional] : when omitted no preselection is applied
    vertex = cms.PSet(
      src    = cms.InputTag("offlinePrimaryVertices"),
      select = cms.string('abs(x)<1. & abs(y)<1. & abs(z)<20. & tracksSize>3 & !isFake')
    )                                       
  ), 
  ## ------------------------------------------------------
  ## SELECTION
  ##
  ## monitor histrograms are filled after each selection
  ## step, the selection is applied in the order defined
  ## by this vector
  ## [mandatory] : may be empty or contain an arbitrary
  ## number of PSets
  selection = cms.VPSet(
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
      select = cms.string("pt>20 & abs(eta)<2.1 & isGlobalMuon & innerTrack.numberOfValidHits>10 & globalTrack.normalizedChi2>-1 & globalTrack.normalizedChi2<10 & (isolationR03.sumPt+isolationR03.emEt+isolationR03.hadEt)/pt<0.1"),       
      min    = cms.int32(1),
      max    = cms.int32(1),
    ),
    cms.PSet(
      label  = cms.string("jets/calo:step2"),
      src    = cms.InputTag("ak5CaloJets"),
      jetCorrector = cms.string("ak5CaloL2L3"),
      select = cms.string("pt>30 & abs(eta)<4.5 & emEnergyFraction>0.01"),
      jetID  = cms.PSet(
        label  = cms.InputTag("ak5JetID"),
        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
      ),
      min = cms.int32(1),
    ), 
    cms.PSet(
      label  = cms.string("jets/calo:step3"),
      src    = cms.InputTag("ak5CaloJets"),
      jetCorrector = cms.string("ak5CaloL2L3"),
      select = cms.string("pt>30 & abs(eta)<4.5 & emEnergyFraction>0.01"),
      jetID  = cms.PSet(
        label  = cms.InputTag("ak5JetID"),
        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
      ),
      min = cms.int32(2),
    ), 
  )
)
