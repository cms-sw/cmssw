
from L1Trigger.Configuration.L1DummyConfig_cff import *

L1GctConfigProducers.CalibrationStyle = cms.string('PowerSeries')
L1GctConfigProducers.PowerSeriesCoefficients = cms.PSet(
        nonTauJetCalib0 = cms.vdouble( 1.0 ),
        nonTauJetCalib1 = cms.vdouble( 1.0 ),
        nonTauJetCalib2 = cms.vdouble( 1.0 ),
        nonTauJetCalib3 = cms.vdouble( 1.0 ),
        nonTauJetCalib4 = cms.vdouble( 1.0 ),
        nonTauJetCalib5 = cms.vdouble( 1.0 ),
        nonTauJetCalib6 = cms.vdouble( 1.0 ),
        nonTauJetCalib7 = cms.vdouble( 1.0 ),
        nonTauJetCalib8 = cms.vdouble( 1.0 ),
        nonTauJetCalib9 = cms.vdouble( 1.0 ),
        nonTauJetCalib10 = cms.vdouble( 1.0 ),
        tauJetCalib0 = cms.vdouble( 1.0 ),
        tauJetCalib1 = cms.vdouble( 1.0 ),
        tauJetCalib2 = cms.vdouble( 1.0 ),
        tauJetCalib3 = cms.vdouble( 1.0 ),
        tauJetCalib4 = cms.vdouble( 1.0 ),
        tauJetCalib5 = cms.vdouble( 1.0 ),
        tauJetCalib6 = cms.vdouble( 1.0 ),
)
