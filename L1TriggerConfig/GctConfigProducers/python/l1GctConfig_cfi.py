import FWCore.ParameterSet.Config as cms

L1GctConfigProducers = cms.ESProducer("L1GctConfigProducers",
    JetFinderCentralJetSeed = cms.double(0.0),
    JetFinderForwardJetSeed = cms.double(0.0),
    TauIsoEtThreshold = cms.double(2.0),
    HtJetEtThreshold = cms.double(10.0),
    MHtJetEtThreshold = cms.double(10.0),
    RctRegionEtLSB = cms.double(0.25),
    GctHtLSB = cms.double(0.25),
    ConvertEtValuesToEnergy = cms.bool(False),

    # energy sum eta ranges
    MEtEtaMask = cms.uint32(0),
    TEtEtaMask = cms.uint32(0),
    MHtEtaMask = cms.uint32(0),
    HtEtaMask = cms.uint32(0),
                                      
    # The CalibrationStyle should be "None", "PiecewiseCubic", "Simple" or "PF"
    # "PowerSeries", "ORCAStyle" are also available, but not recommended
    CalibrationStyle = cms.string('None'),
    PFCoefficients = cms.PSet(
        nonTauJetCalib0 = cms.vdouble(1.25,-1.159,4.032,1.035,0.7912,0.794),
        nonTauJetCalib1 = cms.vdouble(1.277,-1.311,4.379,1.091,0.8263,0.7937),
        nonTauJetCalib2 = cms.vdouble(1.26,-0.9407,4.996,1.12,0.871,0.822),
        nonTauJetCalib3 = cms.vdouble(1.228,-0.7634,5.268,1.197,0.8273,0.8573),
        nonTauJetCalib4 = cms.vdouble(1.055,1.398,3.102,0.917,1.03,0.9293),
        nonTauJetCalib5 = cms.vdouble(1.068,1.459,3.798,0.9516,1.042,0.8555),
        nonTauJetCalib6 = cms.vdouble(0.1593,12.59,6.099,0.2624,3.77,0.9449),
        nonTauJetCalib7 = cms.vdouble(1.421,0.9935,2.861,0.2331,2.625,0.8273),
        nonTauJetCalib8 = cms.vdouble(1.218,1.32,2.861,0.2108,0.7652,-0.1943),
        nonTauJetCalib9 = cms.vdouble(1.155,1.167,1.955,8.466,9.25,0.1372),
        nonTauJetCalib10 = cms.vdouble(1.247,0.414,1.443,0.5022,1.849,0.7142),
        tauJetCalib0 = cms.vdouble(1.25,-1.159,4.032,1.035,0.7912,0.794),
        tauJetCalib1 = cms.vdouble(1.277,-1.311,4.379,1.091,0.8263,0.7937),
        tauJetCalib2 = cms.vdouble(1.26,-0.9407,4.996,1.12,0.871,0.822),
        tauJetCalib3 = cms.vdouble(1.228,-0.7634,5.268,1.197,0.8273,0.8573),
        tauJetCalib4 = cms.vdouble(1.055,1.398,3.102,0.917,1.03,0.9293),
        tauJetCalib5 = cms.vdouble(1.068,1.459,3.798,0.516,1.042,0.8555),
        tauJetCalib6 = cms.vdouble(0.1593,12.59,6.099,0.2624,3.77,0.9449),
    ),
    PiecewiseCubicCoefficients = cms.PSet(
        nonTauJetCalib0 = cms.vdouble(130, 0., 0., 0., 0., 30, 13.2533, 0.690992, -0.00526974, 2.05007e-05, 8, -39.7517, 6.77935, -0.247233, 0.00328425),
        nonTauJetCalib1 = cms.vdouble(130, 0., 0., 0., 0., 30, 14.6191, 0.664683, -0.00480135, 1.84435e-05, 8, -40.8265, 7.06214, -0.265678, 0.00365957),
        nonTauJetCalib2 = cms.vdouble(130, 0., 0., 0., 0., 25, 10.4548, 0.9378, -0.00861004, 3.60848e-05, 8, -51.2717, 9.06628, -0.372669, 0.00554335),
        nonTauJetCalib3 = cms.vdouble(130, 0., 0., 0., 0., 25, 10.0813, 1.1611, -0.0115036, 5.13563e-05, 8.5, -70.2151, 12.3859, -0.56344, 0.00930814),
        nonTauJetCalib4 = cms.vdouble(130, 0., 0., 0., 0., 30, 16.2122, 0.910263, -0.00752007, 2.92421e-05, 8.5, -53.2541, 9.10951, -0.3529, 0.00500446),
        nonTauJetCalib5 = cms.vdouble(130, 0., 0., 0., 0., 20, 11.5901, 1.00117, -0.00939815, 3.85389e-05, 8.5, -79.4911, 15.0639, -0.765169, 0.0140553),
        nonTauJetCalib6 = cms.vdouble(130, 0., 0., 0., 0., 20, 11.3408, 0.670967, -0.00545793, 2.1702e-05, 8, -68.8283, 13.7013, -0.741858, 0.014287),
        nonTauJetCalib7 = cms.vdouble(75, 0., 0., 0., 0., 20, -4.54419, 1.4931, -0.0286778, 0.000261939, 2, -1.69251, 1.70427, -0.0705605, 0.00147167),
        nonTauJetCalib8 = cms.vdouble(70, 0., 0., 0., 0., 20, -1.05904, 0.769109, -0.0118932, 0.00011106, 2, -0.0755721, 0.795891, -0.0222608, 0.000439548),
        nonTauJetCalib9 = cms.vdouble(55, 0., 0., 0., 0., 55, 0.559568, 0.660698, -0.0108648, 0.000117172, 2, 0.559568, 0.660698, -0.0108648, 0.000117172),
        nonTauJetCalib10 = cms.vdouble(30, 0., 0., 0., 0., 30, -0.835944, 1.58517, -0.0650923, 0.00115477, 2, -0.835944, 1.58517, -0.0650923, 0.00115477),
        tauJetCalib0 = cms.vdouble(130, 0., 0., 0., 0., 30, 13.2533, 0.690992, -0.00526974, 2.05007e-05, 8, -39.7517, 6.77935, -0.247233, 0.00328425),
        tauJetCalib1 = cms.vdouble(130, 0., 0., 0., 0., 30, 14.6191, 0.664683, -0.00480135, 1.84435e-05, 8, -40.8265, 7.06214, -0.265678, 0.00365957),
        tauJetCalib2 = cms.vdouble(130, 0., 0., 0., 0., 25, 10.4548, 0.9378, -0.00861004, 3.60848e-05, 8, -51.2717, 9.06628, -0.372669, 0.00554335),
        tauJetCalib3 = cms.vdouble(130, 0., 0., 0., 0., 25, 10.0813, 1.1611, -0.0115036, 5.13563e-05, 8.5, -70.2151, 12.3859, -0.56344, 0.00930814),
        tauJetCalib4 = cms.vdouble(130, 0., 0., 0., 0., 30, 16.2122, 0.910263, -0.00752007, 2.92421e-05, 8.5, -53.2541, 9.10951, -0.3529, 0.00500446),
        tauJetCalib5 = cms.vdouble(130, 0., 0., 0., 0., 20, 11.5901, 1.00117, -0.00939815, 3.85389e-05, 8.5, -79.4911, 15.0639, -0.765169, 0.0140553),
        tauJetCalib6 = cms.vdouble(130, 0., 0., 0., 0., 20, 11.3408, 0.670967, -0.00545793, 2.1702e-05, 8, -68.8283, 13.7013, -0.741858, 0.014287)
    )

)


