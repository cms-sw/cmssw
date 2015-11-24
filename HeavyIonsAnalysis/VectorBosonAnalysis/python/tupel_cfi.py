import FWCore.ParameterSet.Config as cms

tupel = cms.EDAnalyzer("Tupel",
  trigger      = cms.InputTag( "patTrigger" ),                                                               
#  triggerEvent = cms.InputTag( "patTriggerEvent" ),
  #triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),                                       
#  photonSrc   = cms.untracked.InputTag("patPhotons"),
  vtxSrc     = cms.untracked.InputTag("hiSelectedVertex"),
  electronSrc = cms.untracked.InputTag("slimmedElectrons"),
  muonSrc     = cms.untracked.InputTag("patMuons"),
  #tauSrc      = cms.untracked.InputTag("slimmedPatTaus"),                                                    
  jetSrc      = cms.untracked.InputTag("akVs4CalopatJetsWithBtagging"),
  metSrc      = cms.untracked.InputTag("patMETsPF"),
  genSrc      = cms.untracked.InputTag("prunedGenParticles"),
  gjetSrc       = cms.untracked.InputTag('ak4HiGenJets'),
  muonMatch    = cms.string( 'muonTriggerMatchHLTMuons' ),
  muonMatch2    = cms.string( 'muonTriggerMatchHLTMuons2' ),
  elecMatch    = cms.string( 'elecTriggerMatchHLTElecs' ),
  mSrcRho      = cms.untracked.InputTag('fixedGridRhoFastjetAll'),#arbitrary rho now                          
#  CalojetLabel = cms.untracked.InputTag('slimmedJets'), #same collection now BB                             
  CalojetLabel = cms.untracked.InputTag('akVs4CalopatJets'), #same collection now BB                               

  metSource = cms.VInputTag("slimmedMETs","slimmedMETs","slimmedMETs","slimmedMETs"), #no MET corr yet        
  lheSource=cms.untracked.InputTag('source')

)
