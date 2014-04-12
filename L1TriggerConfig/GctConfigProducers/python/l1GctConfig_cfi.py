import FWCore.ParameterSet.Config as cms

L1GctConfigProducers = cms.ESProducer("L1GctConfigProducers",
    JetFinderCentralJetSeed = cms.double(0.0),
    JetFinderForwardJetSeed = cms.double(0.0),
    TauIsoEtThreshold = cms.double(2.0),
    HtJetEtThreshold = cms.double(10.0),
    MHtJetEtThreshold = cms.double(10.0),
    RctRegionEtLSB = cms.double(0.5),
    GctHtLSB = cms.double(0.5),
    ConvertEtValuesToEnergy = cms.bool(False),

    # energy sum eta ranges
    MEtEtaMask = cms.uint32(0x3c000f),
    TEtEtaMask = cms.uint32(0x3c000f),
    MHtEtaMask = cms.uint32(0x3c000f),
    HtEtaMask = cms.uint32(0x3c000f),
                                      
    # The CalibrationStyle should be "None", "PiecewiseCubic", "Simple" or "PF"
    # "PowerSeries", "ORCAStyle" are also available, but not recommended
    CalibrationStyle = cms.string('PF'),
    PFCoefficients = cms.PSet(
           nonTauJetCalib0 = cms.vdouble(1.114,2.297,5.959,1.181,0.7286,0.3673  ),
           nonTauJetCalib1 = cms.vdouble(0.7842,4.331,2.672,0.5743,0.8811,0.4085),
           nonTauJetCalib2 = cms.vdouble(0.961,2.941,2.4,1.248,0.666,0.1041     ),
           nonTauJetCalib3 = cms.vdouble(0.6318,6.6,3.21,0.8551,0.9786,0.291    ),
           nonTauJetCalib4 = cms.vdouble(0.3456,8.992,3.165,0.5798,2.146,0.4912 ),
           nonTauJetCalib5 = cms.vdouble(0.8501,3.892,2.466,1.236,0.8323,0.1809 ),
           nonTauJetCalib6 = cms.vdouble(0.9027,2.581,1.453,1.029,0.6767,-0.1476),
           nonTauJetCalib7 = cms.vdouble(1.117,2.382,1.769,0.0,-1.306,-0.4741   ),
           nonTauJetCalib8 = cms.vdouble(1.634,-1.01,0.7184,1.639,0.6727,-0.2129),
           nonTauJetCalib9 = cms.vdouble(0.9862,3.138,4.672,2.362,1.55,-0.7154  ),
           nonTauJetCalib10 = cms.vdouble(1.245,1.103,1.919,0.3054,5.745,0.8622 ),
           tauJetCalib0 = cms.vdouble(1.114,2.297,5.959,1.181,0.7286,0.3673  ),
           tauJetCalib1 = cms.vdouble(0.7842,4.331,2.672,0.5743,0.8811,0.4085),
           tauJetCalib2 = cms.vdouble(0.961,2.941,2.4,1.248,0.666,0.1041     ),
           tauJetCalib3 = cms.vdouble(0.6318,6.6,3.21,0.8551,0.9786,0.291    ),
           tauJetCalib4 = cms.vdouble(0.3456,8.992,3.165,0.5798,2.146,0.4912 ),
           tauJetCalib5 = cms.vdouble(0.8501,3.892,2.466,1.236,0.8323,0.1809 ),
           tauJetCalib6 = cms.vdouble(0.9027,2.581,1.453,1.029,0.6767,-0.1476),
    )

)


