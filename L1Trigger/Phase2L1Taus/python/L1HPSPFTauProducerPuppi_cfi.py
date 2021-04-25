import FWCore.ParameterSet.Config as cms

L1HPSPFTauProducerPuppi = cms.EDProducer("L1HPSPFTauProducer",
  srcL1PFCands              = cms.InputTag("l1pfCandidates:Puppi"),                                      
  srcL1Jets               = cms.InputTag("Phase1L1TJetProducer:UncalibratedPhase1L1TJetFromPfCandidates"),
  srcL1Vertices             = cms.InputTag("L1TkPrimaryVertex"),
  useChargedPFCandSeeds     = cms.bool(True),                          
  minSeedChargedPFCandPt  = cms.double(5.),
  maxSeedChargedPFCandEta = cms.double(2.4),
  maxSeedChargedPFCandDz  = cms.double(1.e+3),
  useJetSeeds             = cms.bool(True),                          
  minSeedJetPt          = cms.double(30.),
  maxSeedJetEta         = cms.double(2.4),
  signalConeSize            = cms.string("2.8/max(1., pt)"),
  minSignalConeSize        = cms.double(0.05),
  maxSignalConeSize        = cms.double(0.10),
  useStrips                 = cms.bool(True),                                           
  stripSizeEta             = cms.double(0.05),
  stripSizePhi             = cms.double(0.20),
  isolationConeSize         = cms.double(0.4),
  minPFTauPt              = cms.double(20.),
  maxPFTauEta             = cms.double(2.4),                                       
  minLeadChargedPFCandPt  = cms.double(1.),
  maxLeadChargedPFCandEta = cms.double(2.4),
  maxLeadChargedPFCandDz  = cms.double(1.e+3),
  maxChargedIso            = cms.double(1.e+3),
  maxChargedRelIso         = cms.double(1.0),
  deltaRCleaning           = cms.double(0.4),
  signalQualityCuts = cms.PSet(
    chargedHadron = cms.PSet(
      minPt = cms.double(0.),
      maxDz = cms.double(1.e+3),                                          
    ),
    neutralHadron = cms.PSet(
      minPt = cms.double(0.)
    ),                                        
    muon = cms.PSet(
      minPt = cms.double(0.),
      maxDz = cms.double(1.e+3),                                          
    ),
    electron = cms.PSet(
      minPt = cms.double(0.),
      maxDz = cms.double(1.e+3),                                          
    ),                                            
    photon = cms.PSet(
      minPt = cms.double(0.)
    )                                      
  ),
  isolationQualityCuts = cms.PSet(
    chargedHadron = cms.PSet(
      minPt = cms.double(0.),
      maxDz = cms.double(1.e+3),                                          
    ),
    neutralHadron = cms.PSet(
      minPt = cms.double(0.)
    ),                                        
    muon = cms.PSet(
      minPt = cms.double(0.),
      maxDz = cms.double(1.e+3),                                          
    ),
    electron = cms.PSet(
      minPt = cms.double(0.),
      maxDz = cms.double(1.e+3),                                          
    ),                                            
    photon = cms.PSet(
      minPt = cms.double(0.)
    )              
  ),
  applyPreselection = cms.bool(False),                                             
  debug = cms.untracked.bool(False)                                  
)
