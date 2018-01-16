import FWCore.ParameterSet.Config as cms

looseMuonCut = " (isGlobalMuon || isTrackerMuon) && isPFMuon"
looseIsoCut  = "(pfIsolationR04.sumChargedHadronPt + max(0., pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - 0.5 * pfIsolationR04.sumPUPt) ) / pt < 0.2"

tightMuonCut = " isGlobalMuon && isPFMuon && globalTrack.normalizedChi2 < 10. && globalTrack.hitPattern.numberOfValidMuonHits > 0 && " + \
               "numberOfMatchedStations > 1 && innerTrack.hitPattern.numberOfValidPixelHits > 0 && innerTrack.hitPattern.trackerLayersWithMeasurement > 8"
               # CB PV cut!
tightIsoCut  = "(pfIsolationR04.sumChargedHadronPt + max(0., pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - 0.5 * pfIsolationR04.sumPUPt) ) / pt < 0.12"

EletightIsoCut  = "(pfIsolationVariables.sumChargedHadronPt + max(0., pfIsolationVariables.sumNeutralHadronEt + pfIsolationVariables.sumPhotonEt - 0.5 * pfIsolationVariables.sumPUPt) ) / pt < 0.1"
ElelooseIsoCut  = "(pfIsolationVariables.sumChargedHadronPt + max(0., pfIsolationVariables.sumNeutralHadronEt + pfIsolationVariables.sumPhotonEt - 0.5 * pfIsolationVariables.sumPUPt) ) / pt < 0.15"

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
topSingleLeptonDQM_miniAOD = DQMEDAnalyzer('TopSingleLeptonDQM_miniAOD',
  setup = cms.PSet(
    directory = cms.string("Physics/Top/TopSingleLeptonDQM_miniAOD/"),
    sources = cms.PSet(

      elecs = cms.InputTag("slimmedElectrons"),
      jets  = cms.InputTag("slimmedJets"),
      mets  = cms.VInputTag("slimmedMETs", "slimmedMETsNoHF", "slimmedMETsPuppi"),
  
      pvs   = cms.InputTag("offlineSlimmedPrimaryVertices")
    ),
    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),
    pvExtras = cms.PSet(
    ),

    elecExtras = cms.PSet(                                                                                 
      select = cms.string("pt>15 & abs(eta)<2.5 & abs(gsfTrack.d0)<1 & abs(gsfTrack.dz)<20"),
      isolation = cms.string(ElelooseIsoCut),
    ),
    muonExtras = cms.PSet(
      select = cms.string(looseMuonCut + " && pt>10 & abs(eta)<2.4"),
      isolation = cms.string(looseIsoCut),
    ),
    jetExtras = cms.PSet(
      select = cms.string("pt>30 & abs(eta)<2.5 "),
    ),
    massExtras = cms.PSet(
      lowerEdge = cms.double( 70.),
      upperEdge = cms.double(110.)
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

    vertex = cms.PSet(
      src    = cms.InputTag("offlineSlimmedPrimaryVertices")#,
    )                                        
  ),  
  selection = cms.VPSet(
    cms.PSet(
      label  = cms.string("jets:step0"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(2),
    ),
  )
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
topSingleMuonLooseDQM_miniAOD = DQMEDAnalyzer('TopSingleLeptonDQM_miniAOD',

  setup = cms.PSet(
    directory = cms.string("Physics/Top/TopSingleMuonLooseDQM_miniAOD/"),
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
    pvExtras = cms.PSet(
    ),
    muonExtras = cms.PSet(
      select = cms.string(looseMuonCut + " && pt > 10 & abs(eta)<2.4"),
      isolation = cms.string(looseIsoCut)                                               
    ),
    jetExtras = cms.PSet(
      select = cms.string("pt>30 & abs(eta)<2.5"),
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
        )
      ),
    ),
    massExtras = cms.PSet(
      lowerEdge = cms.double( 70.),
      upperEdge = cms.double(110.)
    ),

    triggerExtras = cms.PSet(
      src   = cms.InputTag("TriggerResults","","HLT"),
      paths = cms.vstring(['HLT_Mu3:HLT_QuadJet15U',
                           'HLT_Mu5:HLT_QuadJet15U',
                           'HLT_Mu7:HLT_QuadJet15U',
                           'HLT_Mu9:HLT_QuadJet15U',
                           'HLT_Mu11:HLT_QuadJet15U'])
    )
  ),
  preselection = cms.PSet(
    vertex = cms.PSet(
      src    = cms.InputTag("offlineSlimmedPrimaryVertices")#,
    )
  ),
  selection = cms.VPSet(
    cms.PSet(
      label  = cms.string("muons:step0"),
      src    = cms.InputTag("slimmedMuons"),
      select = cms.string(looseMuonCut + looseIsoCut + " && pt>10 & abs(eta)<2.4"), # CB what about iso? CD Added looseIso
      min    = cms.int32(1),
    ),
    cms.PSet(
      label  = cms.string("jets:step1"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(1),                                               
    ), 
    cms.PSet(
      label  = cms.string("jets:step2"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(2),                                               
    ), 
    cms.PSet(
      label  = cms.string("jets:step3"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(3),                                               
    ), 
    cms.PSet(
      label  = cms.string("jets:step4"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(4),                                               
    ), 
  )
)
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
topSingleMuonMediumDQM_miniAOD = DQMEDAnalyzer('TopSingleLeptonDQM_miniAOD',
  setup = cms.PSet(
    directory = cms.string("Physics/Top/TopSingleMuonMediumDQM_miniAOD/"),
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
    pvExtras = cms.PSet(
    ),
    muonExtras = cms.PSet(
      select    = cms.string(looseMuonCut + " && pt>20 & abs(eta)<2.1"),  
      isolation = cms.string(looseIsoCut)
    ),
    jetExtras = cms.PSet(                       
      select = cms.string("pt>30 & abs(eta)<2.5 "),
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
        )
      ),                                                
    ),
    massExtras = cms.PSet(
      lowerEdge = cms.double( 70.),
      upperEdge = cms.double(110.)
    ),
  ),
  preselection = cms.PSet(
    vertex = cms.PSet(
      src    = cms.InputTag("offlineSlimmedPrimaryVertices")#,
    )
  ),
  selection = cms.VPSet(
    cms.PSet(
      label  = cms.string("muons:step0"),
      src    = cms.InputTag("slimmedMuons"),
      select = cms.string(tightMuonCut +"&&"+ tightIsoCut + " && pt>20 & abs(eta)<2.1"), # CB what about iso? CD Added tightIso      
      min    = cms.int32(1),
      max    = cms.int32(1),
    ),
    cms.PSet(
      label  = cms.string("jets:step1"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(1),
    ), 
    cms.PSet(
      label  = cms.string("jets:step2"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(2),
    ), 
    cms.PSet(
      label  = cms.string("jets:step3"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(3),                                                
    ), 
    cms.PSet(
      label  = cms.string("jets:step4"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(4),                                                
    ),
  )
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
topSingleElectronLooseDQM_miniAOD = DQMEDAnalyzer('TopSingleLeptonDQM_miniAOD',
  setup = cms.PSet(
    directory = cms.string("Physics/Top/TopSingleElectronLooseDQM_miniAOD/"),
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
    pvExtras = cms.PSet(
    ),
    elecExtras = cms.PSet(
      select     = cms.string("pt>20 & abs(eta)<2.5"),
      isolation  = cms.string(ElelooseIsoCut),                                                   
    ),
    jetExtras = cms.PSet(
      select = cms.string("pt>30 & abs(eta)<2.5 "),
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
        )
      ),
    ),
    massExtras = cms.PSet(
      lowerEdge = cms.double( 70.),
      upperEdge = cms.double(110.)
    ),
    triggerExtras = cms.PSet(
      src   = cms.InputTag("TriggerResults","","HLT"),
      paths = cms.vstring(['HLT_Ele15_LW_L1R:HLT_QuadJetU15'])
    )
  ),
  preselection = cms.PSet(
    vertex = cms.PSet(
      src    = cms.InputTag("offlineSlimmedPrimaryVertices")#,
    )
  ),
  selection = cms.VPSet(
    cms.PSet(
      label  = cms.string("elecs:step0"),
      src    = cms.InputTag("slimmedElectrons"),
      select = cms.string("pt>20 & abs(eta)<2.5 && "+ElelooseIsoCut),
      min    = cms.int32(1),
    ),
    cms.PSet(
      label  = cms.string("jets:step1"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(1),                                                   
    ), 
    cms.PSet(
      label  = cms.string("jets:step2"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(2),
    ), 
    cms.PSet(
      label  = cms.string("jets:step3"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(3),
    ), 
    cms.PSet(
      label  = cms.string("jets:step4"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(4),
    ),
  )
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
topSingleElectronMediumDQM_miniAOD = DQMEDAnalyzer('TopSingleLeptonDQM_miniAOD',
  setup = cms.PSet(

    directory = cms.string("Physics/Top/TopSingleElectronMediumDQM_miniAOD/"),
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
    pvExtras = cms.PSet(
    ),
    elecExtras = cms.PSet(
      select     = cms.string("pt>20 & abs(eta)<2.5"),
      isolation  = cms.string(ElelooseIsoCut),
    ),
    jetExtras = cms.PSet(
      select = cms.string("pt>30 & abs(eta)<2.5 "),
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
        )
      ),
    ),
    massExtras = cms.PSet(
      lowerEdge = cms.double( 70.),
      upperEdge = cms.double(110.)
    ),
  ),
  preselection = cms.PSet(
    vertex = cms.PSet(
      src    = cms.InputTag("offlineSlimmedPrimaryVertices")#,
    )
  ),
  selection = cms.VPSet(
    cms.PSet(
      label = cms.string("elecs:step0"),
      src   = cms.InputTag("slimmedElectrons"),
      select = cms.string("pt>30 & abs(eta)<2.5 & abs(gsfTrack.d0)<0.02 & gsfTrack.hitPattern().numberOfLostHits('MISSING_INNER_HITS') <= 0 & (abs(superCluster.eta) <= 1.4442 || abs(superCluster.eta) >= 1.5660) & " + EletightIsoCut),
      min = cms.int32(1),
      max = cms.int32(1),
    ),
    cms.PSet(
      label = cms.string("jets:step1"),
      src   = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(1),
    ), 
    cms.PSet(
      label  = cms.string("jets:step2"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(2),
    ), 
    cms.PSet(
      label  = cms.string("jets:step3"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(3),
    ), 
    cms.PSet(
      label  = cms.string("jets:step4"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.5 "),
      min = cms.int32(4),
    ),
  )
)
