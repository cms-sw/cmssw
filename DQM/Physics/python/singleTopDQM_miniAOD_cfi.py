import FWCore.ParameterSet.Config as cms

EletightIsoCut  = "(pfIsolationVariables.sumChargedHadronPt + max(0., pfIsolationVariables.sumNeutralHadronEt + pfIsolationVariables.sumPhotonEt - 0.5 * pfIsolationVariables.sumPUPt) ) / pt < 0.1"
ElelooseIsoCut  = "(pfIsolationVariables.sumChargedHadronPt + max(0., pfIsolationVariables.sumNeutralHadronEt + pfIsolationVariables.sumPhotonEt - 0.5 * pfIsolationVariables.sumPUPt) ) / pt < 0.15"


singleTopTChannelLeptonDQM_miniAOD = DQMStep1Module('SingleTopTChannelLeptonDQM_miniAOD',

  setup = cms.PSet(
 
    directory = cms.string("Physics/Top/SingleTopDQM_miniAOD/"),

    sources = cms.PSet(
      muons = cms.InputTag("slimmedMuons"),
      elecs = cms.InputTag("slimmedElectrons"),
      jets  = cms.InputTag("slimmedJets"),
      mets  = cms.VInputTag("slimmedMETs", "slimmedMETsNoHF", "slimmedMETsPuppi"),
      pvs   = cms.InputTag("offlineSlimmedPrimaryVertices")
    ),

    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),

    elecExtras = cms.PSet(
                                                                        
      select = cms.string("pt>15 & abs(eta)<2.5 & abs(gsfTrack.d0)<1 & abs(gsfTrack.dz)<20"),
 
      isolation = cms.string(ElelooseIsoCut),
    ),

    muonExtras = cms.PSet(
                                                                               
      select = cms.string("pt>10 & abs(eta)<2.1 & isGlobalMuon & abs(globalTrack.d0)<1 & abs(globalTrack.dz)<20"),
 
    ),
 
    jetExtras = cms.PSet(

      select = cms.string("pt>15 & abs(eta)<2.5 & emEnergyFraction>0.01"),
    ),

    triggerExtras = cms.PSet(
      src   = cms.InputTag("TriggerResults","","HLT"),
      paths = cms.vstring(['HLT_Mu3:HLT_QuadJet15U',
                           'HLT_Mu5:HLT_QuadJet15U',
                           'HLT_Mu7:HLT_QuadJet15U',
                           'HLT_Mu9:HLT_QuadJet15U'])
    )                                            
  ),                                  

  preselection = cms.PSet(

  ),  

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

singleTopMuonMediumDQM_miniAOD = DQMStep1Module('SingleTopTChannelLeptonDQM_miniAOD',

    setup = cms.PSet(

    directory = cms.string("Physics/Top/SingleTopMuonMediumDQM_miniAOD/"),

    sources = cms.PSet(
    muons = cms.InputTag("slimmedMuons"),
    elecs_gsf = cms.InputTag("slimmedElectrons"),
    elecs = cms.InputTag("slimmedElectrons"),
    jets  = cms.InputTag("slimmedJets"),
    mets  = cms.VInputTag("slimmedMETs", "slimmedMETsNoHF", "slimmedMETsPuppi"),
    pvs   = cms.InputTag("offlineSlimmedPrimaryVertices")
    ),

    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),

    muonExtras = cms.PSet(  
 
      select    = cms.string("abs(eta)<2.1")

    ),

    jetExtras = cms.PSet(

      select = cms.string("pt>15 & abs(eta)<2.5"), # & neutralEmEnergyFraction >0.01 & chargedEmEnergyFraction>0.01"),
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

  ),

  preselection = cms.PSet(

  ),

  selection = cms.VPSet(
   cms.PSet(
      label  = cms.string("presel"),
      src    = cms.InputTag("offlineSlimmedPrimaryVertices"),

     
   ),
   cms.PSet(
      label  = cms.string("muons:step0"),
      src    = cms.InputTag("slimmedMuons"),
      select = cms.string("pt>20 & abs(eta)<2.1 &  isGlobalMuon & isTrackerMuon & innerTrack.numberOfValidHits>10 & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10 & innerTrack.hitPattern.pixelLayersWithMeasurement>=1 &  numberOfMatches>1 & abs(innerTrack.dxy)<0.02 & (pfIsolationR04.sumChargedHadronPt + pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt)/pt < 0.15"),

      min    = cms.int32(1),
      max    = cms.int32(1),
    ),
    cms.PSet(
      label  = cms.string("jets:step1"),
      src    = cms.InputTag("slimmedJets"),

      select = cms.string(" pt>30 & abs(eta)<4.5 & numberOfDaughters>1 & ((abs(eta)>2.4) || ( chargedHadronEnergyFraction > 0 & chargedMultiplicity>0 & chargedEmEnergyFraction<0.99)) & neutralEmEnergyFraction < 0.99 & neutralHadronEnergyFraction < 0.99"), 

      min = cms.int32(1),
      max = cms.int32(1),
    ), 
    cms.PSet(
     label  = cms.string("jets:step2"),
     src    = cms.InputTag("slimmedJets"),

     select = cms.string(" pt>30 & abs(eta)<4.5 & numberOfDaughters>1 & ((abs(eta)>2.4) || ( chargedHadronEnergyFraction > 0 & chargedMultiplicity>0 & chargedEmEnergyFraction<0.99)) & neutralEmEnergyFraction < 0.99 & neutralHadronEnergyFraction < 0.99"),
     
     min = cms.int32(2),
     max = cms.int32(2),
    )
  )
)

singleTopElectronMediumDQM_miniAOD = DQMStep1Module('SingleTopTChannelLeptonDQM_miniAOD',

  setup = cms.PSet(
 
    directory = cms.string("Physics/Top/SingleTopElectronMediumDQM_miniAOD/"),

    sources = cms.PSet(
      muons = cms.InputTag("slimmedMuons"),
      elecs_gsf = cms.InputTag("slimmedElectrons"),
      elecs = cms.InputTag("slimmedElectrons"),
      jets  = cms.InputTag("slimmedJets"),
      mets  = cms.VInputTag("slimmedMETs", "slimmedMETsNoHF", "slimmedMETsPuppi"),
      pvs   = cms.InputTag("offlineSlimmedPrimaryVertices")

    ),

    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),

    elecExtras = cms.PSet(
 
      select     = cms.string("pt>25"),
 

    ),

    jetExtras = cms.PSet(

      select = cms.string("pt>15 & abs(eta)<2.5"), 
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
  ),

  preselection = cms.PSet(
 
  ),

  selection = cms.VPSet(
   cms.PSet(
      label  = cms.string("presel"),
      src    = cms.InputTag("offlineSlimmedPrimaryVertices"),
   ),
   cms.PSet(
      label = cms.string("elecs:step0"),
      src   = cms.InputTag("slimmedElectrons"),
      select = cms.string("pt>30 & abs(eta)<2.5 & abs(gsfTrack.d0)<0.02 && gsfTrack.hitPattern().numberOfLostHits('MISSING_INNER_HITS') <= 0 && (abs(superCluster.eta) <= 1.4442 || abs(superCluster.eta) >= 1.5660) && " + EletightIsoCut),
      min = cms.int32(1),
      max = cms.int32(1),
    ),
    cms.PSet(
      label = cms.string("jets:step1"),
      src   = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<4.5 & numberOfDaughters>1 & ((abs(eta)>2.4) || ( chargedHadronEnergyFraction > 0 & chargedMultiplicity>0 & chargedEmEnergyFraction<0.99)) & neutralEmEnergyFraction < 0.99 & neutralHadronEnergyFraction < 0.99"), 

      min = cms.int32(1),
      max = cms.int32(1),
      
    ),
    cms.PSet(
      label = cms.string("jets:step2"),
      src   = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<4.5 & numberOfDaughters>1 & ((abs(eta)>2.4) || ( chargedHadronEnergyFraction > 0 & chargedMultiplicity>0 & chargedEmEnergyFraction<0.99)) & neutralEmEnergyFraction < 0.99 & neutralHadronEnergyFraction < 0.99"),

      min = cms.int32(2),
      max = cms.int32(2),

    ),
  )
)
