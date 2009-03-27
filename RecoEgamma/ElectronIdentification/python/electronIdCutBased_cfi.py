import FWCore.ParameterSet.Config as cms

eidCutBased = cms.EDFilter("EleIdCutBasedRef",
    
    filter = cms.bool(False),
    threshold = cms.double(0.5),

    src = cms.InputTag("gsfElectrons"),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    algorithm = cms.string('eIDCB'),

    #electronIDType can be robust or classbased
    #electronQuality can be loose,tight or highenergy but for the classedbased 
    #electronVersion can be retrived from the PSet name, no version means the last updated 
    electronIDType  = cms.string('robust'),
    electronQuality = cms.string('loose'),
    electronVersion = cms.string(''),

    # variables H/E sigmaietaieta deltaphiin deltaetain e2x5/e5X5 e1x5/e5x5 (barrel/endcap)
    #Robust Loose Cuts
    #V00 CMSSW16X optimization 
    #V01 CMSSW22X optimization 
    robustlooseEleIDCutsV00 = cms.PSet(
        barrel = cms.vdouble(0.115, 0.014, 0.09, 0.009,-1,-1),
        endcap = cms.vdouble(0.15, 0.0275, 0.092, 0.0105,-1,-1)
    ),
    robustlooseEleIDCutsV01 = cms.PSet(
       barrel = cms.vdouble(0.075, 0.0132, 0.058, 0.077,-1,-1),
       endcap = cms.vdouble(0.083, 0.027, 0.042, 0.01,-1,-1)
    ),
    robustlooseEleIDCuts = cms.PSet(
       barrel = cms.vdouble(0.075, 0.0132, 0.058, 0.077,-1,-1),
       endcap = cms.vdouble(0.083, 0.027, 0.042, 0.01,-1,-1)
    ),
    
    #Robust Tight Cuts
    #V00 CMSSW16X optimization 
    #V01 CMSSW22X optimization 
    robusttightEleIDCutsV00 = cms.PSet(
        barrel = cms.vdouble(0.015, 0.0092, 0.020, 0.0025,-1,-1),
        endcap = cms.vdouble(0.018, 0.025, 0.020, 0.0040,-1,-1)
    ),
    robusttightEleIDCuts = cms.PSet(
        barrel = cms.vdouble(0.015, 0.0092, 0.020, 0.0025,-1,-1),
        endcap = cms.vdouble(0.018, 0.025, 0.020, 0.0040,-1,-1)
    ),
   
    #Robust High Energy Cuts
    #V00 CMSSW16X optimization 
    #V01 CMSSW22X optimization 
    robusthighenergyEleIDCutsV00 = cms.PSet(
        barrel = cms.vdouble(0.050, 0.011, 0.090, 0.005,-1,-1),
        endcap = cms.vdouble(0.100, 0.0275, 0.090, 0.007,-1,-1)
    ),
    robusthighenergyEleIDCutsV01 = cms.PSet(
        barrel = cms.vdouble(0.050, -1, 0.090, 0.005,0.94,0.83),
        endcap = cms.vdouble(0.050, 0.0275, 0.090, 0.007,-1,-1)
    ),
    robusthighenergyEleIDCuts = cms.PSet(
        barrel = cms.vdouble(0.050, -1, 0.090, 0.005,0.94,0.83),
        endcap = cms.vdouble(0.050, 0.0275, 0.090, 0.007,-1,-1)
    ),

    #Class Based Loose Cuts
    #V00 CMSSW16X optimization 
    #V01 CMSSW22X optimization 
    classbasedlooseEleIDCutsV00 = cms.PSet(
        deltaEtaIn   = cms.vdouble(0.009, 0.0045, 0.0085, 0.0, 0.0105, 0.0068, 0.01, 0.0),
        deltaPhiIn   = cms.vdouble(0.05, 0.025, 0.053, 0.09, 0.07, 0.03, 0.092, 0.092),
        eSeedOverPin = cms.vdouble(0.11, 0.91, 0.11, 0.0, 0.0, 0.85, 0.0, 0.0),
        hOverE       = cms.vdouble(0.115, 0.1, 0.055, 0.0, 0.145, 0.12, 0.15, 0.0),
        sigmaEtaEta  = cms.vdouble(0.014, 0.012, 0.0115, 0.0, 0.0275, 0.0265, 0.0265, 0.0)
        ),
    classbasedlooseEleIDCutsV01 = cms.PSet(
        deltaEtaIn   = cms.vdouble (0.0078, 0.00259, 0.0062, 0.0, 0.0078, 0.0061, 0.0061, 0.0),
        deltaPhiIn   = cms.vdouble (0.053, 0.0189, 0.059, 0.099, 0.0278, 0.0157, 0.042, 0.080),
        eSeedOverPin = cms.vdouble (0.30, 0.92, 0.211, 0.0, 0.42, 0.88, 0.68, 0.0),
        hOverE       = cms.vdouble (0.076, 0.033, 0.070, 0.0, 0.083, 0.0148, 0.033, 0.0),
        sigmaEtaEta  = cms.vdouble (0.0101, 0.0095, 0.0097, 0.0, 0.0271, 0.0267, 0.0259, 0.0)
        ),
    classbasedlooseEleIDCuts = cms.PSet(
        deltaEtaIn   = cms.vdouble (0.0078, 0.00259, 0.0062, 0.0, 0.0078, 0.0061, 0.0061, 0.0),
        deltaPhiIn   = cms.vdouble (0.053, 0.0189, 0.059, 0.099, 0.0278, 0.0157, 0.042, 0.080),
        eSeedOverPin = cms.vdouble (0.30, 0.92, 0.211, 0.0, 0.42, 0.88, 0.68, 0.0),
        hOverE       = cms.vdouble (0.076, 0.033, 0.070, 0.0, 0.083, 0.0148, 0.033, 0.0),
        sigmaEtaEta  = cms.vdouble (0.0101, 0.0095, 0.0097, 0.0, 0.0271, 0.0267, 0.0259, 0.0)
        ),

    #Class Based Tight Cuts
    #V00 CMSSW16X optimization 
    #V01 CMSSW22X optimization 
    classbasedtightEleIDCutsV00 = cms.PSet(
        deltaEtaIn   = cms.vdouble(0.0055, 0.003, 0.0065, 0.0, 0.006, 0.0055, 0.0075, 0.0),
        deltaPhiIn   = cms.vdouble(0.032, 0.016, 0.0525, 0.09, 0.025, 0.035, 0.065, 0.092),
	     eSeedOverPin = cms.vdouble(0.24, 0.94, 0.11, 0.0, 0.32, 0.83, 0.0, 0.0),
        hOverE       = cms.vdouble(0.05, 0.042, 0.045, 0.0, 0.055, 0.037, 0.05, 0.0),
        sigmaEtaEta  = cms.vdouble(0.0125, 0.011, 0.01, 0.0, 0.0265, 0.0252, 0.026, 0.0)
    ),
    classbasedtightEleIDCutsV01 = cms.PSet(
        deltaEtaIn   = cms.vdouble (0.0043, 0.00282, 0.0036, 0.0, 0.0066, 0.0049, 0.0041, 0.0),
        deltaPhiIn   = cms.vdouble (0.0225, 0.0114, 0.0234, 0.039, 0.0215, 0.0095, 0.0148, 0.0167),
        eSeedOverPin = cms.vdouble (0.32, 0.94, 0.221, 0.0, 0.74, 0.89, 0.66, 0.0),
        hOverE       = cms.vdouble (0.056, 0.0221, 0.037, 0.0, 0.0268, 0.0102, 0.0104, 0.0),
        sigmaEtaEta  = cms.vdouble (0.0095, 0.0094, 0.0094, 0.0, 0.0260, 0.0257, 0.0246, 0.0)
    ),   
    classbasedtightEleIDCuts = cms.PSet(
        deltaEtaIn   = cms.vdouble (0.0043, 0.00282, 0.0036, 0.0, 0.0066, 0.0049, 0.0041, 0.0),
        deltaPhiIn   = cms.vdouble (0.0225, 0.0114, 0.0234, 0.039, 0.0215, 0.0095, 0.0148, 0.0167),
        eSeedOverPin = cms.vdouble (0.32, 0.94, 0.221, 0.0, 0.74, 0.89, 0.66, 0.0),
        hOverE       = cms.vdouble (0.056, 0.0221, 0.037, 0.0, 0.0268, 0.0102, 0.0104, 0.0),
        sigmaEtaEta  = cms.vdouble (0.0095, 0.0094, 0.0094, 0.0, 0.0260, 0.0257, 0.0246, 0.0)
    )   
)
