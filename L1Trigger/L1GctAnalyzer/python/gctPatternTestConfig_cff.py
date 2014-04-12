#
# This configuration fragment sets the physics parameters in the GCT emulator to the current internal pattern test key. 
# In the future the global tag could be used with more up to date internal key and this hack can be removed.
#
# Alex Tapper 8/9/10
#

from L1Trigger.Configuration.L1DummyConfig_cff import *

# RCT EM
RCTConfigProducers.eGammaLSB = cms.double(0.25)
l1CaloScales.L1CaloEmEtScaleLSB = cms.double(0.25)
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

# RCT Regions
RCTConfigProducers.jetMETLSB = cms.double(0.25)
l1CaloScales.L1CaloRegionEtScaleLSB = cms.double(0.25)

# GCT jet finding params
L1GctConfigProducers.JetFinderCentralJetSeed = cms.double(0.0)
L1GctConfigProducers.JetFinderForwardJetSeed = cms.double(0.0)
L1GctConfigProducers.RctRegionEtLSB = cms.double(0.25)
L1GctConfigProducers.GctHtLSB = cms.double(0.25)
L1GctConfigProducers.TauIsoEtThreshold = cms.double(2.0)
L1GctConfigProducers.HtJetEtThreshold = cms.double(0.25)
L1GctConfigProducers.MHtJetEtThreshold = cms.double(0.25)

# GCT jet corrections
L1GctConfigProducers.CalibrationStyle = cms.string('PowerSeries')
L1GctConfigProducers.PowerSeriesCoefficients = cms.PSet(
            nonTauJetCalib0 = cms.vdouble( 0.0 ),
                    nonTauJetCalib1 = cms.vdouble( 0.0 ),
                    nonTauJetCalib2 = cms.vdouble( 0.0 ),
                    nonTauJetCalib3 = cms.vdouble( 0.0 ),
                    nonTauJetCalib4 = cms.vdouble( 0.0 ),
                    nonTauJetCalib5 = cms.vdouble( 0.0 ),
                    nonTauJetCalib6 = cms.vdouble( 0.0 ),
                    nonTauJetCalib7 = cms.vdouble( 0.0 ),
                    nonTauJetCalib8 = cms.vdouble( 0.0 ),
                    nonTauJetCalib9 = cms.vdouble( 0.0 ),
                    nonTauJetCalib10 = cms.vdouble( 0.0 ),
                    tauJetCalib0 = cms.vdouble( 0.0 ),
                    tauJetCalib1 = cms.vdouble( 0.0 ),
                    tauJetCalib2 = cms.vdouble( 0.0 ),
                    tauJetCalib3 = cms.vdouble( 0.0 ),
                    tauJetCalib4 = cms.vdouble( 0.0 ),
                    tauJetCalib5 = cms.vdouble( 0.0 ),
                    tauJetCalib6 = cms.vdouble( 0.0 ),
            )

# GCT jet output scale
l1CaloScales.L1CaloJetThresholds = cms.vdouble(
        0.0, 2.0, 4.0, 6.0, 8.0,
            10.0, 12.0, 14.0, 16.0, 18.0,
            20.0, 22.0, 24.0, 26.0, 28.0,
            30.0, 32.0, 34.0, 36.0, 38.0,
            40.0, 42.0, 44.0, 46.0, 48.0,
            50.0, 52.0, 54.0, 56.0, 58.0,
            60.0, 62.0, 64.0, 66.0, 68.0,
            70.0, 72.0, 74.0, 76.0, 78.0,
            80.0, 82.0, 84.0, 86.0, 88.0,
            90.0, 92.0, 94.0, 96.0, 98.0,
            100.0, 102.0, 104.0, 106.0, 108.0,
            110.0, 112.0, 114.0, 116.0, 118.0,
            120.0, 122.0, 124.0, 126.0)

# GCT HF ring sum scale
l1CaloScales.L1HfRingThresholds = cms.vdouble(0.0, 2.0, 3.0, 4.0, 6.0, 50., 200., 500.)
