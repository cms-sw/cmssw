import FWCore.ParameterSet.Config as cms

# module to produce jet correction factors associated in a valuemap
jetCorrFactors = cms.EDProducer("JetCorrFactorsProducer",
     useEMF    = cms.bool(False),                         ## the use of emf in the JEC
                                                          ## is not yet implemented here 
     jetSource = cms.InputTag("iterativeCone5CaloJets"),  ## input collection of jets 

     ## tags for the jet correctors; when not available the string should be set to 'none' 
     L1Offset  = cms.string('none'),
     L2Relative= cms.string('Summer08_L2Relative_IC5Calo'),
     L3Absolute= cms.string('Summer08Redigi_L3Absolute_IC5Calo'),
     L4EMF     = cms.string('none'),
     L5Flavor  = cms.string('L5Flavor_fromQCD_iterativeCone5'),
     L6UE      = cms.string('none'),                           
     L7Parton  = cms.string('L7parton_IC5_080921')
)


