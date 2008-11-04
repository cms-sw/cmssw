import FWCore.ParameterSet.Config as cms

# module to produce jet correction factors associated in a valuemap
jetCorrFactors = cms.EDProducer("JetCorrFactorsProducer",
     jetSource = cms.InputTag("iterativeCone5CaloJets"),
 
     L1JetCorrector      = cms.string('none'),
     
     L2JetCorrector      = cms.string('L2RelativeJetCorrectorIC5Calo'),
     
     L3JetCorrector      = cms.string('L3AbsoluteJetCorrectorIC5Calo'),

     L4JetCorrector      = cms.string('none'),
     
     L5udsJetCorrector   = cms.string('none'),
     L5gluonJetCorrector = cms.string('none'),
     L5cJetCorrector     = cms.string('none'),
     L5bJetCorrector     = cms.string('none'),
     
     L6JetCorrector      = cms.string('none'),
                           
     L7udsJetCorrector   = cms.string('L7PartonJetCorrectorIC5qJet'),
     L7gluonJetCorrector = cms.string('L7PartonJetCorrectorIC5gJet'),
     L7cJetCorrector     = cms.string('L7PartonJetCorrectorIC5cJet'),
     L7bJetCorrector     = cms.string('L7PartonJetCorrectorIC5bJet')
  )


