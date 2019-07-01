import FWCore.ParameterSet.Config as cms

classbasedlooseEleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutdetain = cms.vdouble(
1.37e-02, 6.78e-03, 2.41e-02, 1.87e-02, 1.61e-02, 2.24e-02, 2.52e-02, 3.08e-02, 2.73e-02
),
cutdetainl = cms.vdouble(
1.24e-02, 5.03e-03, 2.57e-02, 2.28e-02, 1.18e-02, 1.78e-02, 1.88e-02, 1.40e-01, 2.40e-02
),
cutdphiin = cms.vdouble(
8.97e-02, 2.62e-01, 3.53e-01, 1.16e-01, 3.57e-01, 3.19e-01, 3.42e-01, 4.04e-01, 3.36e-01
),
cutdphiinl = cms.vdouble(
7.47e-02, 2.50e-01, 3.56e-01, 9.56e-02, 3.47e-01, 3.26e-01, 3.33e-01, 6.47e-01, 2.89e-01
),
cuteseedopcor = cms.vdouble(
6.30e-01, 8.20e-01, 4.01e-01, 7.18e-01, 4.00e-01, 4.58e-01, 1.50e-01, 6.64e-01, 3.73e-01
),
cutfmishits = cms.vdouble(
4.50e+00, 1.50e+00, 1.50e+00, 2.50e+00, 2.50e+00, 1.50e+00, 4.50e+00, 3.50e+00, 3.50e+00
),
cuthoe = cms.vdouble(
2.47e-01, 1.37e-01, 1.47e-01, 3.71e-01, 5.88e-02, 1.47e-01, 5.20e-01, 4.52e-01, 4.04e-01
),
cuthoel = cms.vdouble(
2.36e-01, 1.26e-01, 1.47e-01, 3.75e-01, 3.92e-02, 1.45e-01, 3.65e-01, 3.83e-01, 3.84e-01
),
cutip_gsf = cms.vdouble(
5.51e-02, 7.65e-02, 1.43e-01, 8.74e-02, 5.94e-01, 3.70e-01, 9.13e-02, 1.15e+00, 2.31e-01
),
cutip_gsfl = cms.vdouble(
1.86e-02, 7.59e-02, 1.38e-01, 4.73e-02, 6.20e-01, 3.04e-01, 1.09e-01, 7.75e-01, 4.79e-02
),
cutiso_sum = cms.vdouble(
3.30e+01, 1.70e+01, 1.79e+01, 1.88e+01, 8.55e+00, 1.25e+01, 1.76e+01, 1.85e+01, 2.98e+00
),
cutiso_sumoet = cms.vdouble(
3.45e+01, 1.27e+01, 1.21e+01, 1.99e+01, 6.35e+00, 8.85e+00, 1.40e+01, 1.05e+01, 9.74e+00
),
cutiso_sumoetl = cms.vdouble(
1.13e+01, 9.05e+00, 9.07e+00, 9.94e+00, 5.25e+00, 6.15e+00, 1.07e+01, 1.08e+01, 4.40e+00
),
cutsee = cms.vdouble(
1.76e-02, 1.25e-02, 1.81e-02, 4.15e-02, 3.64e-02, 4.18e-02, 1.46e-02, 6.78e-02, 1.33e-01
),
cutseel = cms.vdouble(
1.64e-02, 1.18e-02, 1.50e-02, 5.23e-02, 3.26e-02, 4.56e-02, 1.85e-02, 5.89e-02, 5.44e-02
)
)

electronsWithPresel = cms.EDFilter("GsfElectronSelector",
                                   src = cms.InputTag("ecalDrivenGsfElectrons"),
                                   cut = cms.string("pt > 10 && ecalDrivenSeed && passingCutBasedPreselection"),
                                   )

electronsCiCLoose = cms.EDFilter("EleIdCutBased",
                                 src = cms.InputTag("electronsWithPresel"),
                                 algorithm = cms.string("eIDCB"),
                                 threshold = cms.double(14.5),
                                 electronIDType = "classbased",
                                 electronQuality = "loose",
                                 electronVersion = "V06",
                                 additionalCategories = True,
                                 classbasedlooseEleIDCutsV06 = classbasedlooseEleIDCutsV06,
                                 etBinning = cms.bool(False),
                                 version = cms.string(""),
                                 verticesCollection = cms.InputTag('offlinePrimaryVertices'),
                                 reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
                                 reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
                                 )
