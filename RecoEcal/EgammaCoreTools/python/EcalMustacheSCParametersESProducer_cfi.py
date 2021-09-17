import FWCore.ParameterSet.Config as cms

ecalMustacheSCParametersESProducer = cms.ESProducer("EcalMustacheSCParametersESProducer",
    sqrtLogClustETuning = cms.double(1.1),

    # Parameters from the analysis by L. Zygala [https://indico.cern.ch/event/949294/contributions/3988389/attachments/2091573/3514649/2020_08_26_Clustering.pdf]
    # mustache parambola parameters depending on cluster energy and seed crystal eta
    parabolaParameterSets = cms.VPSet(
        # average parameters
        cms.PSet(
            log10EMin = cms.double(-3.),
            etaMin = cms.double(0.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble( -0.00681785, -0.00239516),
            w1Up = cms.vdouble( 0.000699995, -0.00554331),
            w0Low = cms.vdouble(-0.00681785, -0.00239516),
            w1Low = cms.vdouble(0.000699995, -0.00554331)
        )
    )
)

