import FWCore.ParameterSet.Config as cms

looseMuonCut = "muonRef.isNonnull && (muonRef.isGlobalMuon || muonRef.isTrackerMuon) && muonRef.isPFMuon"
looseIsoCut  = "(muonRef.pfIsolationR04.sumChargedHadronPt + max(0., muonRef.pfIsolationR04.sumNeutralHadronEt + muonRef.pfIsolationR04.sumPhotonEt - 0.5 * muonRef.pfIsolationR04.sumPUPt) ) / muonRef.pt < 0.2"

tightMuonCut = "muonRef.isNonnull && muonRef.isGlobalMuon && muonRef.isPFMuon && muonRef.globalTrack.normalizedChi2 < 10. && muonRef.globalTrack.hitPattern.numberOfValidMuonHits > 0 && " + \
               "muonRef.numberOfMatchedStations > 1 && muonRef.innerTrack.hitPattern.numberOfValidPixelHits > 0 && muonRef.innerTrack.hitPattern.trackerLayersWithMeasurement > 8"
               # CB PV cut!
tightIsoCut  = "(muonRef.pfIsolationR04.sumChargedHadronPt + max(0., muonRef.pfIsolationR04.sumNeutralHadronEt + muonRef.pfIsolationR04.sumPhotonEt - 0.5 * muonRef.pfIsolationR04.sumPUPt) ) / muonRef.pt < 0.12"

EletightIsoCut  = "(gsfElectronRef.pfIsolationVariables.sumChargedHadronPt + max(0., gsfElectronRef.pfIsolationVariables.sumNeutralHadronEt + gsfElectronRef.pfIsolationVariables.sumPhotonEt - 0.5 * gsfElectronRef.pfIsolationVariables.sumPUPt) ) / gsfElectronRef.pt < 0.1"
ElelooseIsoCut  = "(gsfElectronRef.pfIsolationVariables.sumChargedHadronPt + max(0., gsfElectronRef.pfIsolationVariables.sumNeutralHadronEt + gsfElectronRef.pfIsolationVariables.sumPhotonEt - 0.5 * gsfElectronRef.pfIsolationVariables.sumPUPt) ) / gsfElectronRef.pt < 0.15"

topSingleLeptonDQM = cms.EDAnalyzer("TopSingleLeptonDQM",
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
    directory = cms.string("Physics/Top/TopSingleLeptonDQM/"),
    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("pfIsolatedMuonsEI"),
      elecs = cms.InputTag("pfIsolatedElectronsEI"),
      jets  = cms.InputTag("ak4PFJetsCHS"),
      mets  = cms.VInputTag("caloMet", "tcMet", "pfMet"),
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
    ## [optional] : when omitted all monitoring plots for electrons
    ## will be filled w/o extras
    elecExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o cut on electronId
      ##electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), cutValue = cms.double(0.5) ),
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates                                                                                            
      select = cms.string("pt>15 & abs(eta)<2.5 & abs(gsfTrack.d0)<1 & abs(gsfTrack.dz)<20"),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot 
      isolation = cms.string(ElelooseIsoCut),
    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras
    muonExtras = cms.PSet(
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates                                                                                            
      select = cms.string(looseMuonCut + " && pt>10 & abs(eta)<2.4"),
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
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets
      select = cms.string("pt>30 & abs(eta)<2.5 "),
    ),
    ## [optional] : when omitted no mass window will be applied
    ## for the W mass befor filling the event monitoring plots
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
                           'HLT_Mu9:HLT_QuadJet15U'])
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
    #trigger = cms.PSet(
    #  src    = cms.InputTag("TriggerResults","","HLT"),
    #  select = cms.vstring(['HLT_Mu11', 'HLT_Ele15_LW_L1R', 'HLT_QuadJet30'])
    #),
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
  ##    
  selection = cms.VPSet(
    cms.PSet(
      label  = cms.string("jets/pf:step0"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
 #     ),
      min = cms.int32(2),
    ),
  )
)

topSingleMuonLooseDQM = cms.EDAnalyzer("TopSingleLeptonDQM",
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
    directory = cms.string("Physics/Top/TopSingleMuonLooseDQM/"),
    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("pfIsolatedMuonsEI"),
      elecs = cms.InputTag("pfIsolatedElectronsEI"),
      jets  = cms.InputTag("ak4PFJetsCHS"),
      mets  = cms.VInputTag("caloMet", "tcMet", "pfMet"),
      pvs   = cms.InputTag("offlinePrimaryVertices")
    ),
    ## [optional] : when omitted the verbosity level is set to STANDARD
    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),
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
      select = cms.string(looseMuonCut + " && pt > 10 & abs(eta)<2.4"),
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot                                                    
      isolation = cms.string(looseIsoCut)                                               
    ),
    ## [optional] : when omitted all monitoring plots for jets
    ## will be filled w/o extras
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets                                               
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID                                                                                                                     
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
  #    ),                                                    
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets                                                
      select = cms.string("pt>30 & abs(eta)<2.5"),
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
        ),
        cvsVertex = cms.PSet(
          label = cms.InputTag("combinedSecondaryVertexBJetTags"),
          workingPoint = cms.double(0.898) 
          # CSV Tight from https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagPerformanceOP#B_tagging_Operating_Points_for_5
        )
      ),
    ),
    ## [optional] : when omitted no mass window will be applied
    ## for the W mass befor filling the event monitoring plots
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
    ## [optional] : when omitted no preselection is applied
    #trigger = cms.PSet(
    #  src    = cms.InputTag("TriggerResults","","HLT"),
    #  select = cms.vstring(['HLT_Mu11'])
    #),
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
      label  = cms.string("muons:step0"),
      src    = cms.InputTag("pfIsolatedMuonsEI"),
      select = cms.string(looseMuonCut + looseIsoCut + " && pt>10 & abs(eta)<2.4"), # CB what about iso? CD Added looseIso
      min    = cms.int32(1),
    ),
    cms.PSet(
      label  = cms.string("jets/pf:step1"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      #select = cms.string("pt>30 & abs(eta)<2.5 & emEnergyFraction>0.01"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
   #   ),
      min = cms.int32(1),                                               
    ), 
    cms.PSet(
      label  = cms.string("jets/pf:step2"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
    #  ),
      min = cms.int32(2),                                               
    ), 
    cms.PSet(
      label  = cms.string("jets/pf:step3"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
     # ),
      min = cms.int32(3),                                               
    ), 
    cms.PSet(
      label  = cms.string("jets/pf:step4"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      min = cms.int32(4),                                               
    ), 
  )
)
topSingleMuonMediumDQM = cms.EDAnalyzer("TopSingleLeptonDQM",
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
    directory = cms.string("Physics/Top/TopSingleMuonMediumDQM/"),
    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("pfIsolatedMuonsEI"),
      elecs = cms.InputTag("pfIsolatedElectronsEI"),
      jets  = cms.InputTag("ak4PFJetsCHS"),
      mets  = cms.VInputTag("caloMet", "tcMet", "pfMet"),
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
      select    = cms.string(looseMuonCut + " && pt>20 & abs(eta)<2.1"),  
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot                                                    
      isolation = cms.string(looseIsoCut)
    ),
    ## [optional] : when omitted all monitoring plots for jets
    ## will be filled w/o extras
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID                                                                                                   
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
 #     ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets                                                
      select = cms.string("pt>30 & abs(eta)<2.5 "),
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
        ),
        cvsVertex = cms.PSet(
          label = cms.InputTag("combinedSecondaryVertexBJetTags"),
          workingPoint = cms.double(0.898) 
          # CSV Tight from https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagPerformanceOP#B_tagging_Operating_Points_for_5
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
#    triggerExtras = cms.PSet(
#      src   = cms.InputTag("TriggerResults","","HLT"),
#     paths = cms.vstring(['HLT_Mu3:HLT_QuadJet15U',
#                          'HLT_Mu5:HLT_QuadJet15U',
#                          'HLT_Mu7:HLT_QuadJet15U',
#                          'HLT_Mu9:HLT_QuadJet15U',
#                          'HLT_Mu11:HLT_QuadJet15U'])      
#    )
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
    #  src    = cms.InputTag("TriggerResults","","HLT"),
    #  select = cms.vstring(['HLT_Mu15_v2'])
    #),
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
      label  = cms.string("muons:step0"),
      src    = cms.InputTag("pfIsolatedMuonsEI"),
      select = cms.string(tightMuonCut +"&&"+ tightIsoCut + " && pt>20 & abs(eta)<2.1"), # CB what about iso? CD Added tightIso      
      min    = cms.int32(1),
      max    = cms.int32(1),
    ),
    cms.PSet(
      label  = cms.string("jets/pf:step1"),
      #src    = cms.InputTag("ak4PFJetsCHS"),
      src    = cms.InputTag("ak4PFJetsCHS"),
#      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      #select = cms.string("pt>30 & abs(eta)<2.5 & emEnergyFraction>0.01"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
  #    ),
      min = cms.int32(1),
    ), 
    cms.PSet(
      label  = cms.string("jets/pf:step2"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
   #   ),
      min = cms.int32(2),
    ), 
    cms.PSet(
      label  = cms.string("jets/pf:step3"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
 #     ),
      min = cms.int32(3),                                                
    ), 
    cms.PSet(
      label  = cms.string("jets/pf:step4"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      min = cms.int32(4),                                                
    ),
  )
)

topSingleElectronLooseDQM = cms.EDAnalyzer("TopSingleLeptonDQM",
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
    directory = cms.string("Physics/Top/TopSingleElectronLooseDQM/"),
    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("pfIsolatedMuonsEI"),
      elecs = cms.InputTag("pfIsolatedElectronsEI"),
      jets  = cms.InputTag("ak4PFJetsCHS"),
      mets  = cms.VInputTag("caloMet", "tcMet", "pfMet"),
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
    ## [optional] : when omitted all monitoring plots for electrons
    ## will be filled w/o extras
    elecExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o cut on electronId
      #electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), cutValue = cms.double(0.0) ),
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates
      select     = cms.string("pt>20 & abs(eta)<2.5"),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot                                                    
      isolation  = cms.string(ElelooseIsoCut),                                                   
    ),
    ## [optional] : when omitted all monitoring plots for jets
    ## will be filled w/o extras
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID                                                   
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
    #  ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets
      #select = cms.string("pt>30 & abs(eta)<2.5 & emEnergyFraction>0.01"), 
      select = cms.string("pt>30 & abs(eta)<2.5 "),
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
        ),
        cvsVertex = cms.PSet(
          label = cms.InputTag("combinedSecondaryVertexBJetTags"),
          workingPoint = cms.double(0.898) 
          # CSV Tight from https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagPerformanceOP#B_tagging_Operating_Points_for_5
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
      paths = cms.vstring(['HLT_Ele15_LW_L1R:HLT_QuadJetU15'])
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
    #trigger = cms.PSet(
    #  src    = cms.InputTag("TriggerResults","","HLT"),
    #  select = cms.vstring(['HLT_Ele15_SW_CaloEleId_L1R'])
    #),
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
      label  = cms.string("elecs:step0"),
      src    = cms.InputTag("pfIsolatedElectronsEI"),
      select = cms.string("pt>20 & abs(eta)<2.5 && "+ElelooseIsoCut),
      min    = cms.int32(1),
    ),
    cms.PSet(
      label  = cms.string("jets/pf:step1"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
     # ),
      min = cms.int32(1),                                                   
    ), 
    cms.PSet(
      label  = cms.string("jets/pf:step2"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      min = cms.int32(2),
    ), 
    cms.PSet(
      label  = cms.string("jets/pf:step3"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
 #     ),
      min = cms.int32(3),
    ), 
    cms.PSet(
      label  = cms.string("jets/pf:step4"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
  #    ),
      min = cms.int32(4),
    ),
  )
)

topSingleElectronMediumDQM = cms.EDAnalyzer("TopSingleLeptonDQM",
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
    directory = cms.string("Physics/Top/TopSingleElectronMediumDQM/"),
    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("pfIsolatedMuonsEI"),
      elecs = cms.InputTag("pfIsolatedElectronsEI"),
      jets  = cms.InputTag("ak4PFJetsCHS"),
      mets  = cms.VInputTag("caloMet", "tcMet", "pfMet"),
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
    ## [optional] : when omitted all monitoring plots for electrons
    ## will be filled w/o extras
    elecExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o cut on electronId
      #electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), cutValue = cms.double(0.0) ),
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates
      select     = cms.string("pt>20 & abs(eta)<2.5"),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot 
      isolation  = cms.string(ElelooseIsoCut),
    ),
    ## [optional] : when omitted all monitoring plots for jets
    ## will be filled w/o extras
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
   #   ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets 
      select = cms.string("pt>30 & abs(eta)<2.5 "),
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
        ),
        cvsVertex = cms.PSet(
          label = cms.InputTag("combinedSecondaryVertexBJetTags"),
          workingPoint = cms.double(0.898) 
          # CSV Tight from https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagPerformanceOP#B_tagging_Operating_Points_for_5
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
    #triggerExtras = cms.PSet(
    #  src   = cms.InputTag("TriggerResults","","HLT"),
    #  paths = cms.vstring([ 'HLT_Ele15_LW_L1R:HLT_QuadJetU15'])
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
    #  src    = cms.InputTag("TriggerResults","","HLT"),
    #  select = cms.vstring(['HLT_Ele15_SW_CaloEleId_L1R'])
    #),
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
      label = cms.string("elecs:step0"),
      src   = cms.InputTag("pfIsolatedElectronsEI"),
      select = cms.string("pt>30 & abs(eta)<2.5 & abs(gsfElectronRef.gsfTrack.d0)<0.02 & gsfElectronRef.gsfTrack.hitPattern().numberOfHits('MISSING_INNER_HITS') <= 0 & (abs(gsfElectronRef.superCluster.eta) <= 1.4442 || abs(gsfElectronRef.superCluster.eta) >= 1.5660) & " + EletightIsoCut),
      min = cms.int32(1),
      max = cms.int32(1),
    ),
    cms.PSet(
      label = cms.string("jets/pf:step1"),
      src   = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      min = cms.int32(1),
    ), 
    cms.PSet(
      label  = cms.string("jets/pf:step2"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
 #     ),
      min = cms.int32(2),
    ), 
    cms.PSet(
      label  = cms.string("jets/pf:step3"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
  #    ),
      min = cms.int32(3),
    ), 
    cms.PSet(
      label  = cms.string("jets/pf:step4"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      #jetID  = cms.PSet(
        #label  = cms.InputTag("ak5JetID"),
        #select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
   #   ),
      min = cms.int32(4),
    ),
  )
)
