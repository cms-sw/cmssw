
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

l1CaloScales.L1CaloEmEtScaleLSB = cms.double(0.25)
l1CaloScales.L1CaloRegionEtScaleLSB = cms.double(0.25)
l1CaloScales.L1CaloEmThresholds = cms.vdouble(
        0.0, 0.5, 1.0, 1.5, 2.0, 
        2.5, 3.0, 3.5, 4.0, 4.5,
        5.0, 5.5, 6.0, 6.5, 7.0, 
        7.5, 8.0, 8.5, 9.0, 9.5,
        10.0, 10.5, 11.0, 11.5, 12.0, 
        12.5, 13.0, 13.5, 14.0, 14.5,
        15.0, 15.5, 16.0, 16.5, 17.0, 
        17.5, 18.0, 18.5, 19.0, 19.5,
        20.0, 20.5, 21.0, 21.5, 22.0, 
        22.5, 23.0, 23.5, 24.0, 24.5,
        25.0, 25.5, 26.0, 26.5, 27.0, 
        27.5, 28.0, 28.5, 29.0, 29.5,
        30.0, 30.5, 31.0, 31.5)
l1CaloScales.L1CaloJetThresholds = cms.vdouble(
        0.0, 1.0, 2.0, 3.0, 4.0, 
        5.0, 6.0, 7.0, 8.0, 9.0, 
        10.0, 11.0, 12.0, 13.0, 14.0, 
        15.0, 16.0, 17.0, 18.0, 19.0, 
        20.0, 21.0, 22.0, 23.0, 24.0, 
        25.0, 26.0, 27.0, 28.0, 29.0, 
        30.0, 31.0, 32.0, 33.0, 34.0, 
        35.0, 36.0, 37.0, 38.0, 39.0, 
        40.0, 41.0, 42.0, 43.0, 44.0, 
        45.0, 46.0, 47.0, 48.0, 49.0, 
        50.0, 51.0, 52.0, 53.0, 54.0, 
        55.0, 56.0, 57.0, 58.0, 59.0, 
        60.0, 61.0, 62.0, 63.0)

