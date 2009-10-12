import FWCore.ParameterSet.Config as cms

# module to produce jet correction factors associated in a valuemap
jetCorrFactors = cms.EDProducer("JetCorrFactorsProducer",
     ## the use of emf in the JEC is not yet implemented
     useEMF     = cms.bool(False),
     ## input collection of jets
     jetSource  = cms.InputTag("ak5CaloJets"),
     ## tags for the jet correctors; when not available the
     ## string should be set to 'none'                                 
     L1Offset   = cms.string('none'),
#    L2Relative = cms.string('Summer09_L2Relative_AK5Calo'),
     L2Relative = cms.string('Summer09_7TeV_L2Relative_IC5Calo'), ## this is still IC to consistent with the CMSSW_3_1_4 plain
#    L3Absolute = cms.string('Summer09_L3Absolute_AK5Calo'),
     L3Absolute = cms.string('Summer09_7TeV_L3Absolute_IC5Calo'), ## this is still IC to consistent with the CMSSW_3_1_4 plain
     L4EMF      = cms.string('none'),
     L5Flavor   = cms.string('L5Flavor_IC5'),  # to be changed to L5Flavor   = cms.string('L5Flavor_AK5'),
     L6UE       = cms.string('none'),
     L7Parton   = cms.string('L7Parton_SC5'),  # to be changed to L7Parton   = cms.string('L7Parton_AK5'),
     ## choose sample type for flavor dependend corrections:
     sampleType = cms.string('dijet')  ##  'dijet': from dijet sample
                                       ##  'ttbar': from ttbar sample

)


