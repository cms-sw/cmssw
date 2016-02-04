import FWCore.ParameterSet.Config as cms

# module to produce jet correction factors associated in a valuemap
patJetCorrFactors = cms.EDProducer("JetCorrFactorsProducer",
    ## the use of emf in the JEC is not yet implemented
    emf = cms.bool(False),
    ## input collection of jets
    src = cms.InputTag("ak5CaloJets"),
    ## payload postfix for testing
    payload = cms.string('AK5Calo'),
    ## correction levels
    levels = cms.vstring(
        ## tags for the individual jet corrections; when
        ## not available the string should be set to 'none'    
        'L2Relative', 'L3Absolute', 'L5Flavor', 'L7Parton'
    ), 
    flavorType = cms.string('J') ## alternatively use 'T'
)
