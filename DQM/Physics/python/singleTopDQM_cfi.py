import FWCore.ParameterSet.Config as cms

EletightIsoCut  = "(gsfElectronRef.pfIsolationVariables.sumChargedHadronPt + max(0., gsfElectronRef.pfIsolationVariables.sumNeutralHadronEt + gsfElectronRef.pfIsolationVariables.sumPhotonEt - 0.5 * gsfElectronRef.pfIsolationVariables.sumPUPt) ) / gsfElectronRef.pt < 0.1"
ElelooseIsoCut  = "(gsfElectronRef.pfIsolationVariables.sumChargedHadronPt + max(0., gsfElectronRef.pfIsolationVariables.sumNeutralHadronEt + gsfElectronRef.pfIsolationVariables.sumPhotonEt - 0.5 * gsfElectronRef.pfIsolationVariables.sumPUPt) ) / gsfElectronRef.pt < 0.15"


singleTopTChannelLeptonDQM = cms.EDAnalyzer("SingleTopTChannelLeptonDQM",
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
    directory = cms.string("Physics/Top/SingleTopDQM/"),
    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("pfIsolatedMuonsEI"),
      elecs = cms.InputTag("pfIsolatedElectronsEI"),
      jets  = cms.InputTag("ak4PFJetsCHS"),
      mets  = cms.VInputTag("met", "tcMet", "pfMetEI"),
      pvs   = cms.InputTag("offlinePrimaryVertices")
    ),
    ## [optional] : when omitted the verbosity level is set to STANDARD
    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),
    ## [optional] : when omitted all monitoring plots for primary vertices
    ## will be filled w/o extras
#    pvExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the primary vertex candidates                                                                                            
#      select = cms.string("abs(x)<1. & abs(y)<1. & abs(z)<20. & tracksSize>3 & !isFake")
#    ),
    ## [optional] : when omitted all monitoring plots for electrons
    ## will be filled w/o extras
    elecExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o cut on electronId
      ##electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), cutValue = cms.double(0.5) ),  
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates                                                                                            
      select = cms.string("pt>15 & abs(eta)<2.5 & abs(gsfElectronRef.gsfTrack.d0)<1 & abs(gsfElectronRef.gsfTrack.dz)<20"),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot 
      isolation = cms.string(ElelooseIsoCut),
    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras
    muonExtras = cms.PSet(
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates                                                                                            
      select = cms.string("pt>10 & abs(eta)<2.1 & isGlobalMuon & abs(globalTrack.d0)<1 & abs(globalTrack.dz)<20"),
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot                                                    
#      isolation = cms.string("(isolationR03.sumPt+isolationR03.emEt+isolationR03.hadEt)/pt<0.1"),
    ),
    ## [optional] : when omitted all monitoring plots for jets will
    ## be filled from uncorrected jets
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets                                            
      jetCorrector = cms.string("ak4CaloL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID                                                   
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak4JetID"),
#        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#      ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets
      select = cms.string("pt>15 & abs(eta)<2.5 & emEnergyFraction>0.01"),
    ),
    ## [optional] : when omitted no mass window will be applied
    ## for the W mass befor filling the event monitoring plots
#    massExtras = cms.PSet(
#      lowerEdge = cms.double( 70.),
#      upperEdge = cms.double(110.)
#    ),
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
#    trigger = cms.PSet(
#      src    = cms.InputTag("TriggerResults","","HLT"),
#      select = cms.vstring(['HLT_Mu11', 'HLT_Ele15_LW_L1R', 'HLT_QuadJet30'])
#    ),
    ## [optional] : when omitted no preselection is applied
#    vertex = cms.PSet(
#      src    = cms.InputTag("offlinePrimaryVertices"),
#      select = cms.string('abs(x)<1. & abs(y)<1. & abs(z)<20. & tracksSize>3 & !isFake')
#    )                                        
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
      label  = cms.string("jets/calo:step0"),
      src    = cms.InputTag("ak4CaloJets"),
      select = cms.string("pt>20 & abs(eta)<2.1 & 0.05<emEnergyFraction"),
      jetID  = cms.PSet(
        label  = cms.InputTag("ak4JetID"),
        select = cms.string("fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
      ),
      min = cms.int32(2),
    )
  )
)

singleTopMuonMediumDQM = cms.EDAnalyzer("SingleTopTChannelLeptonDQM",
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
    directory = cms.string("Physics/Top/SingleTopMuonMediumDQM/"),
    ## [mandatory]
    sources = cms.PSet(
    muons = cms.InputTag("pfIsolatedMuonsEI"),
    elecs_gsf = cms.InputTag("gedGsfElectrons"),
    elecs = cms.InputTag("pfIsolatedElectronsEI"),
    jets  = cms.InputTag("ak4PFJetsCHS"),
    mets  = cms.VInputTag("met", "tcMet", "pfMetEI"),
    pvs   = cms.InputTag("offlinePrimaryVertices")
    ),
    ## [optional] : when omitted the verbosity level is set to STANDARD
    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),
    ## [optional] : when omitted all monitoring plots for primary vertices
    ## will be filled w/o extras
#    pvExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the primary vertex candidates                                                                                            
#      select = cms.string("") #abs(x)<1. & abs(y)<1. & abs(z)<20. & tracksSize>3 & !isFake")
#    ),
    ## [optional] : when omitted all monitoring plots for muons
    ## will be filled w/o extras                                           
    muonExtras = cms.PSet(  
      ## when omitted muon plots will be filled w/o additional pre-
      ## selection of the muon candidates 
      select    = cms.string("abs(muonRef.eta)<2.1")
      ## & isGlobalMuon & innerTrack.numberOfValidHits>10 & globalTrack.normalizedChi2>-1 & globalTrack.normalizedChi2<10
      ##& (isolationR03.sumPt+isolationR03.emEt+isolationR03.hadEt)/pt<0.1"),  
      ## when omitted isolated muon multiplicity plot will be equi-
      ## valent to inclusive muon multiplicity plot                                                    
   ##   isolation = cms.string("(muonRef.isolationR03.sumPt+muonRef.isolationR03.emEt+muonRef.isolationR03.hadEt)/muonRef.pt<10" )
      ##    isolation = cms.string("(muonRef.isolationR03.sumPt+muonRef.isolationR03.emEt+muonRef.isolationR03.hadEt)/muonRef.pt<0.1")
    ),
    ## [optional] : when omitted all monitoring plots for jets
    ## will be filled w/o extras
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID                                                                                                   
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak4JetID"),
#        select = cms.string(""), ##fHPD < 0.98 & n90Hits>1 & restrictedEMF<1")
#     ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets                                                
      select = cms.string("pt>15 & abs(eta)<2.5"), # & neutralEmEnergyFraction >0.01 & chargedEmEnergyFraction>0.01"),
      ## when omitted monitor histograms for b-tagging will not be filled                                                                                                   
      jetBTaggers  = cms.PSet(
        trackCountingEff = cms.PSet(
          label = cms.InputTag("trackCountingHighEffBJetTags" ),
          workingPoint = cms.double(1.25)
        ),
        trackCountingPur = cms.PSet(
         label = cms.InputTag("trackCountingHighPurBJetTags" ),
          workingPoint = cms.double(3.41)
        ),
        secondaryVertex  = cms.PSet(
          label = cms.InputTag("simpleSecondaryVertexHighEffBJetTags"),
          workingPoint = cms.double(2.05)
        ),
        combinedSecondaryVertex  = cms.PSet(
          label = cms.InputTag("combinedSecondaryVertexBJetTags"),
          workingPoint = cms.double(0.898)
        )
     )                                                
   )
    ## [optional] : when omitted no mass window will be applied
    ## for the W mass before filling the event monitoring plots
#    massExtras = cms.PSet(
#      lowerEdge = cms.double( 70.),
#      upperEdge = cms.double(110.)
#    ),
    ## [optional] : when omitted the monitoring plots for triggering
    ## will be empty
#    triggerExtras = cms.PSet(
#      src   = cms.InputTag("TriggerResults","","HLT"),
#     paths = cms.vstring(['HLT_IsoMu17_eta2p1_CentralPFNoPUJet30_BTagIPIter_v1'])
#                          'HLT_IsoMu24_eta2p1_v12',
#                          'HLT_IsoMu20_eta2p1_CentralPFJet30_BTagIPIter_v2',
#                          'HLT_IsoMu20_eta2p1_CentralPFJet30_BTagIPIter_v3'])      
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
#    trigger = cms.PSet(
#    src    = cms.InputTag("TriggerResults","","HLT"),
#      select = cms.vstring(['HLT_IsoMu17_eta2p1_CentralPFNoPUJet30_BTagIPIter_v1'])
#    ),
    ## [optional] : when omitted no preselection is applied
#    vertex = cms.PSet(
#      src    = cms.InputTag("offlinePrimaryVertices"),
#      select = cms.string('!isFake && ndof >= 4 && abs(z)<24. && position.Rho <= 2.0')
#    )
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
      label  = cms.string("presel"),
      src    = cms.InputTag("offlinePrimaryVertices"),
      select = cms.string('!isFake && ndof >= 4 && abs(z)<24. && position.Rho <= 2.0 '),
     
   ),
   cms.PSet(
      label  = cms.string("muons/pf:step0"),
      src    = cms.InputTag("pfIsolatedMuonsEI"),
      select = cms.string("muonRef.pt>20 & abs(muonRef.eta)<2.1 & muonRef.isNonnull & muonRef.innerTrack.isNonnull & muonRef.isGlobalMuon & muonRef.isTrackerMuon & muonRef.innerTrack.numberOfValidHits>10 & muonRef.globalTrack.hitPattern.numberOfValidMuonHits>0 & muonRef.globalTrack.normalizedChi2<10 & muonRef.innerTrack.hitPattern.pixelLayersWithMeasurement>=1 &  muonRef.numberOfMatches>1 & abs(muonRef.innerTrack.dxy)<0.02 & (muonRef.pfIsolationR04.sumChargedHadronPt + muonRef.pfIsolationR04.sumNeutralHadronEt + muonRef.pfIsolationR04.sumPhotonEt)/muonRef.pt < 0.15"),

      min    = cms.int32(1),
      max    = cms.int32(1),
    ),
    cms.PSet(
      label  = cms.string("jets/pf:step1"),
      src    = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string(" pt>30 & abs(eta)<4.5 & numberOfDaughters>1 & ((abs(eta)>2.4) || ( chargedHadronEnergyFraction > 0 & chargedMultiplicity>0 & chargedEmEnergyFraction<0.99)) & neutralEmEnergyFraction < 0.99 & neutralHadronEnergyFraction < 0.99"), 

      min = cms.int32(1),
      max = cms.int32(1),
    ), 
    cms.PSet(
     label  = cms.string("jets/pf:step2"),
     src    = cms.InputTag("ak4PFJetsCHS"),
     jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
     select = cms.string(" pt>30 & abs(eta)<4.5 & numberOfDaughters>1 & ((abs(eta)>2.4) || ( chargedHadronEnergyFraction > 0 & chargedMultiplicity>0 & chargedEmEnergyFraction<0.99)) & neutralEmEnergyFraction < 0.99 & neutralHadronEnergyFraction < 0.99"),
     
     min = cms.int32(2),
     max = cms.int32(2),
    )
  )
)

singleTopElectronMediumDQM = cms.EDAnalyzer("SingleTopTChannelLeptonDQM",
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
    directory = cms.string("Physics/Top/SingleTopElectronMediumDQM/"),
    ## [mandatory]
    sources = cms.PSet(
      muons = cms.InputTag("pfIsolatedMuonsEI"),
      elecs_gsf = cms.InputTag("gedGsfElectrons"),
      elecs = cms.InputTag("pfIsolatedElectronsEI"),
      jets  = cms.InputTag("ak4PFJetsCHS"),
      mets  = cms.VInputTag("met", "tcMet", "pfMetEI"),
      pvs   = cms.InputTag("offlinePrimaryVertices")

    ),
    ## [optional] : when omitted the verbosity level is set to STANDARD
    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),
    ## [optional] : when omitted all monitoring plots for primary vertices
    ## will be filled w/o extras
#    pvExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the primary vertex candidates                                                                                            
#      select = cms.string("abs(x)<1. & abs(y)<1. & abs(z)<20. & tracksSize>3 & !isFake")
#    ),
    ## [optional] : when omitted all monitoring plots for electrons
    ## will be filled w/o extras
    elecExtras = cms.PSet(
      ## when omitted electron plots will be filled w/o cut on electronId
      ##electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), cutValue = cms.double(0.5) ),  
      ## when omitted electron plots will be filled w/o additional pre-
      ## selection of the electron candidates
      select     = cms.string("pt>25"), ##  & abs(eta)<2.5 & (dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/pt<0.1"),
      ## when omitted isolated electron multiplicity plot will be equi-
      ## valent to inclusive electron multiplicity plot 
     ## isolation  = cms.string(ElelooseIsoCut),

    ),
    ## [optional] : when omitted all monitoring plots for jets
    ## will be filled w/o extras
    jetExtras = cms.PSet(
      ## when omitted monitor plots for pt will be filled from uncorrected
      ## jets
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      ## when omitted monitor plots will be filled w/o additional cut on
      ## jetID
#      jetID  = cms.PSet(
#        label  = cms.InputTag("ak4JetID"),
#        select = cms.string(" ")
#      ),
      ## when omitted no extra selection will be applied on jets before
      ## filling the monitor histograms; if jetCorrector is present the
      ## selection will be applied to corrected jets 
      select = cms.string("pt>15 & abs(eta)<2.5"), ## & emEnergyFraction>0.01"),
      ## when omitted monitor histograms for b-tagging will not be filled
      jetBTaggers  = cms.PSet(
        trackCountingEff = cms.PSet(
          label = cms.InputTag("trackCountingHighEffBJetTags" ),
          workingPoint = cms.double(1.25)
        ),
        trackCountingPur = cms.PSet(
          label = cms.InputTag("trackCountingHighPurBJetTags" ),
          workingPoint = cms.double(3.41)
        ),
        secondaryVertex  = cms.PSet(
          label = cms.InputTag("simpleSecondaryVertexHighEffBJetTags"),
          workingPoint = cms.double(2.05)
        ),
        combinedSecondaryVertex  = cms.PSet(
          label = cms.InputTag("combinedSecondaryVertexBJetTags"),
          workingPoint = cms.double(0.898)
        )
      )
    ),
    ## [optional] : when omitted no mass window will be applied
    ## for the W mass before filling the event monitoring plots
#    massExtras = cms.PSet(
#      lowerEdge = cms.double( 70.),
#      upperEdge = cms.double(110.)
#    ),
    ## [optional] : when omitted the monitoring plots for triggering
    ## will be empty
#    triggerExtras = cms.PSet(
#      src   = cms.InputTag("TriggerResults","","HLT"),
#      paths = cms.vstring([ 'HLT_Ele15_LW_L1R:HLT_QuadJetU15'])
##      paths = cms.vstring([''])
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
#    trigger = cms.PSet(
#     src    = cms.InputTag("TriggerResults","","HLT"),
#     select = cms.vstring(['HLT_Ele15_SW_CaloEleId_L1R'])
#    ),
    ## [optional] : when omitted no preselection is applied
#    vertex = cms.PSet(
#      src    = cms.InputTag("offlinePrimaryVertices"),
#      select = cms.string('!isFake && ndof >= 4 && abs(z)<24. && position.Rho <= 2.0')
#    )
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
      label  = cms.string("presel"),
      src    = cms.InputTag("offlinePrimaryVertices"),
      select = cms.string('!isFake && ndof >= 4 && abs(z)<24. && position.Rho <= 2.0'),
   ),
   cms.PSet(
      label = cms.string("elecs/pf:step0"),
      src   = cms.InputTag("pfIsolatedElectronsEI"),
##      electronId = cms.PSet( src = cms.InputTag("mvaTrigV0"), cutValue = cms.double(0.5) ),  
      select = cms.string("pt>30 & abs(eta)<2.5 & abs(gsfElectronRef.gsfTrack.d0)<0.02 && gsfElectronRef.gsfTrack.hitPattern().numberOfHits('MISSING_INNER_HITS') <= 0 && (abs(gsfElectronRef.superCluster.eta) <= 1.4442 || abs(gsfElectronRef.superCluster.eta) >= 1.5660) && " + EletightIsoCut),
      min = cms.int32(1),
      max = cms.int32(1),
    ),
    cms.PSet(
      label = cms.string("jets/pf:step1"),
      src   = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30 & abs(eta)<4.5 & numberOfDaughters>1 & ((abs(eta)>2.4) || ( chargedHadronEnergyFraction > 0 & chargedMultiplicity>0 & chargedEmEnergyFraction<0.99)) & neutralEmEnergyFraction < 0.99 & neutralHadronEnergyFraction < 0.99"), 

      min = cms.int32(1),
      max = cms.int32(1),
      
    ),
    cms.PSet(
      label = cms.string("jets/pf:step2"),
      src   = cms.InputTag("ak4PFJetsCHS"),
      jetCorrector = cms.string("topDQMak5PFCHSL2L3"),
      select = cms.string("pt>30 & abs(eta)<4.5 & numberOfDaughters>1 & ((abs(eta)>2.4) || ( chargedHadronEnergyFraction > 0 & chargedMultiplicity>0 & chargedEmEnergyFraction<0.99)) & neutralEmEnergyFraction < 0.99 & neutralHadronEnergyFraction < 0.99"),

      min = cms.int32(2),
      max = cms.int32(2),

    ),
  )
)
