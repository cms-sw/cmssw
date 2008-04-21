import FWCore.ParameterSet.Config as cms

# prescale factors for L1 GT algorithm triggers 
l1GtPrescaleFactorsAlgoTrig = cms.ESProducer("L1GtPrescaleFactorsAlgoTrigTrivialProducer",
    PrescaleFactors = cms.vint32(4000, 2000, 1, 1, 1, 
        1, 1, 10000, 1000, 100, 
        1, 1, 1, 1, 10000, 
        1000, 100, 100, 1, 1, 
        1, 100000, 100000, 10000, 10000, 
        100, 1, 1, 1, 100000, 
        100000, 10000, 1, 1000, 1000, 
        1, 1, 1000, 100, 1, 
        1, 1, 1, 10000, 1, 
        1, 1, 1, 1, 1, 
        1, 1000, 100, 100, 1, 
        1, 1, 1, 20, 1, 
        1, 1, 1, 1, 20, 
        1, 1, 1, 1, 1, 
        20, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 100, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 10000, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 3000000, 
        3000000, 10000, 5000, 100000, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1)
)


