import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi import *

eidHZZVeryLoose = eidCutBasedExt.clone()
eidHZZVeryLoose.electronIDType = 'classbased'
eidHZZVeryLoose.electronQuality = 'veryloose'
eidHZZVeryLoose.electronVersion = 'V06'
eidHZZVeryLoose.additionalCategories = True
eidHZZVeryLoose.classbasedverylooseEleIDCutsV06 = cms.PSet(
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutip_gsf = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutip_gsfl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdcotdist = cms.vdouble(
3.32e-02,  2.92e-02,  2.49e-02,  3.92e-02,  3.41e-02,  3.96e-02,  2.91e-02,  3.95e-02,  7.71e-03
),
cutdetain = cms.vdouble(
1.28e-02, 6.91e-03, 2.45e-02, 2.41e-02, 9.09e-03, 1.75e-02, 4.19e-02, 3.06e-02, 2.38e-02
),
cutdetainl = cms.vdouble(
1.25e-02, 8.21e-03, 2.52e-02, 2.52e-02, 1.51e-02, 3.56e-02, 1.95e-02, 1.54e-01, 3.44e-02
),
cutdphiin = cms.vdouble(
8.03e-02, 2.72e-01, 3.60e-01, 1.14e-01, 2.89e-01, 3.15e-01, 3.94e-01, 4.04e-01, 7.67e-01
),
cutdphiinl = cms.vdouble(
7.25e-02, 2.69e-01, 3.58e-01, 9.52e-02, 2.84e-01, 3.27e-01, 3.47e-01, 6.50e-01, 2.97e-01
),
cuteseedopcor = cms.vdouble(
6.31e-01, 3.20e-01, 4.04e-01, 7.36e-01, 2.25e-01, 4.56e-01, 2.49e-01, 3.34e-01, 2.12e-01
),
cutfmishits = cms.vdouble(
4.50e+00, 1.50e+00, 1.50e+00, 4.50e+00, 2.50e+00, 2.50e+00, 1.50e+00, 3.50e+00, 3.50e+00
),
cuthoe = cms.vdouble(
2.15e-01, 8.68e-02, 1.47e-01, 3.71e-01, 5.51e-02, 1.46e-01, 4.55e-01, 4.27e-01, 4.05e-01
),
cuthoel = cms.vdouble(
2.14e-01, 1.07e-01, 1.46e-01, 3.71e-01, 5.93e-02, 1.46e-01, 3.53e-01, 3.81e-01, 3.87e-01
),
cutsee = cms.vdouble(
1.70e-02, 1.28e-02, 1.89e-02, 5.23e-02, 3.02e-02, 3.34e-02, 1.35e-02, 6.31e-02, 5.77e-02
),
cutseel = cms.vdouble(
1.70e-02, 1.50e-02, 1.89e-02, 5.11e-02, 4.48e-02, 5.11e-02, 2.00e-02, 5.82e-02, 9.40e-02
)
)

eidHZZLoose = eidCutBasedExt.clone()
eidHZZLoose.electronIDType = 'classbased'
eidHZZLoose.electronQuality = 'loose'
eidHZZLoose.electronVersion = 'V06'
eidHZZLoose.additionalCategories = True
eidHZZLoose.classbasedlooseEleIDCutsV06 = cms.PSet(
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutip_gsf = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutip_gsfl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdcotdist = cms.vdouble(
2.68e-02,  2.36e-02,  2.21e-02,  3.72e-02,  3.17e-02,  3.61e-02,  2.55e-02,  3.75e-02,  2.16e-04
),
cutdetain = cms.vdouble(
1.28e-02, 5.16e-03, 1.26e-02, 2.25e-02, 6.72e-03, 1.12e-02, 1.38e-02, 3.06e-02, 1.52e-02
),
cutdetainl = cms.vdouble(
1.25e-02, 5.74e-03, 2.02e-02, 2.52e-02, 9.95e-03, 1.99e-02, 1.68e-02, 1.54e-01, 2.76e-02
),
cutdphiin = cms.vdouble(
7.63e-02, 2.51e-01, 3.26e-01, 1.13e-01, 2.56e-01, 2.79e-01, 3.43e-01, 4.04e-01, 7.67e-01
),
cutdphiinl = cms.vdouble(
6.97e-02, 2.56e-01, 3.14e-01, 9.48e-02, 2.48e-01, 2.88e-01, 3.29e-01, 6.50e-01, 2.97e-01
),
cuteseedopcor = cms.vdouble(
6.31e-01, 3.57e-01, 4.12e-01, 7.36e-01, 3.10e-01, 4.56e-01, 2.49e-01, 3.34e-01, 2.20e-01
),
cutfmishits = cms.vdouble(
4.50e+00, 1.50e+00, 1.50e+00, 2.50e+00, 2.50e+00, 1.50e+00, 1.50e+00, 2.50e+00, 1.50e+00
),
cuthoe = cms.vdouble(
2.15e-01, 5.50e-02, 1.47e-01, 3.71e-01, 2.92e-02, 9.46e-02, 4.55e-01, 4.27e-01, 4.05e-01
),
cuthoel = cms.vdouble(
2.14e-01, 8.06e-02, 1.46e-01, 3.71e-01, 3.49e-02, 9.26e-02, 3.53e-01, 3.81e-01, 3.32e-01
),
cutsee = cms.vdouble(
1.70e-02, 1.12e-02, 1.22e-02, 4.60e-02, 2.89e-02, 3.06e-02, 1.05e-02, 3.39e-02, 2.96e-02
),
cutseel = cms.vdouble(
1.70e-02, 1.26e-02, 1.69e-02, 5.11e-02, 4.05e-02, 5.11e-02, 1.57e-02, 5.82e-02, 8.76e-02
)
)

eidHZZMedium = eidCutBasedExt.clone()
eidHZZMedium.electronIDType = 'classbased'
eidHZZMedium.electronQuality = 'medium'
eidHZZMedium.electronVersion = 'V06'
eidHZZMedium.additionalCategories = True
eidHZZMedium.classbasedmediumEleIDCutsV06 = cms.PSet(
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutip_gsf = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutip_gsfl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdcotdist = cms.vdouble(
2.11e-02,  1.86e-02,  1.55e-02,  3.40e-02,  2.85e-02,  3.32e-02,  1.64e-02,  3.75e-02,  1.30e-04
),
cutdetain = cms.vdouble(
1.11e-02, 4.50e-03, 7.80e-03, 1.70e-02, 5.93e-03, 9.50e-03, 1.16e-02, 2.80e-02, -5.82e-04
),
cutdetainl = cms.vdouble(
1.13e-02, 4.87e-03, 1.23e-02, 2.36e-02, 7.08e-03, 1.25e-02, 1.27e-02, 8.69e-02, 2.76e-02
),
cutdphiin = cms.vdouble(
5.85e-02, 7.14e-02, 2.69e-01, 8.06e-02, 5.50e-02, 2.35e-01, 2.50e-01, 3.92e-01, 7.67e-01
),
cutdphiinl = cms.vdouble(
5.82e-02, 1.23e-01, 2.61e-01, 8.08e-02, 1.30e-01, 2.26e-01, 2.55e-01, 4.36e-01, 2.51e-01
),
cuteseedopcor = cms.vdouble(
6.98e-01, 8.44e-01, 4.89e-01, 7.89e-01, 4.32e-01, 5.24e-01, 3.21e-01, 7.19e-01, 2.81e-01
),
cutfmishits = cms.vdouble(
3.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 5.00e-01
),
cuthoe = cms.vdouble(
1.56e-01, 4.52e-02, 1.39e-01, 3.41e-01, 2.12e-02, 4.95e-02, 4.13e-01, 4.20e-01, -2.37e-02
),
cuthoel = cms.vdouble(
1.51e-01, 6.50e-02, 1.46e-01, 3.34e-01, 2.43e-02, 5.30e-02, 3.41e-01, 3.80e-01, 2.22e-01
),
cutsee = cms.vdouble(
1.59e-02, 1.09e-02, 1.06e-02, 3.45e-02, 2.83e-02, 2.90e-02, 9.92e-03, 2.71e-02, 1.68e-02
),
cutseel = cms.vdouble(
1.58e-02, 1.19e-02, 1.35e-02, 4.75e-02, 3.43e-02, 4.46e-02, 1.14e-02, 5.82e-02, 7.22e-02
)
)

eidHZZTight = eidCutBasedExt.clone()
eidHZZTight.electronIDType = 'classbased'
eidHZZTight.electronQuality = 'tight'
eidHZZTight.electronVersion = 'V06'
eidHZZTight.additionalCategories = True
eidHZZTight.classbasedtightEleIDCutsV06 = cms.PSet(
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutip_gsf = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutip_gsfl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdcotdist = cms.vdouble(
1.48e-02,  1.50e-02,  8.25e-03,  3.16e-02,  2.85e-02,  3.15e-02,  6.62e-03,  3.48e-02,  3.63e-06
),
cutdetain = cms.vdouble(
1.09e-02,  4.23e-03,  6.82e-03,  1.42e-02,  5.48e-03,  8.72e-03,  1.16e-02,  1.71e-02,  -4.58e-03
),
cutdetainl = cms.vdouble(
1.11e-02,  4.66e-03,  9.98e-03,  1.94e-02,  5.88e-03,  9.24e-03,  1.27e-02,  3.83e-02,  2.54e-02
),
cutdphiin = cms.vdouble(
5.74e-02,  3.56e-02,  2.56e-01,  6.90e-02,  2.46e-02,  2.10e-01,  5.67e-02,  3.65e-01,  3.40e-01
),
cutdphiinl = cms.vdouble(
5.77e-02,  5.55e-02,  2.53e-01,  7.67e-02,  3.45e-02,  2.26e-01,  9.15e-02,  3.72e-01,  2.09e-01
),
cuteseedopcor = cms.vdouble(
7.03e-01,  9.03e-01,  7.35e-01,  8.21e-01,  5.47e-01,  6.53e-01,  3.21e-01,  8.24e-01,  2.81e-01
),
cutfmishits = cms.vdouble(
1.50e+00,  1.50e+00,  5.00e-01,  1.50e+00,  1.50e+00,  5.00e-01,  1.50e+00,  5.00e-01,  5.00e-01
),
cuthoe = cms.vdouble(
9.16e-02,  3.94e-02,  6.34e-02,  3.35e-01,  1.74e-02,  3.55e-02,  3.38e-01,  4.20e-01,  -1.10e-01
),
cuthoel = cms.vdouble(
4.86e-02,  5.70e-02,  6.94e-02,  3.34e-01,  1.81e-02,  3.46e-02,  3.41e-01,  3.80e-01,  2.22e-01
),
cutsee = cms.vdouble(
1.45e-02,  1.09e-02,  1.03e-02,  3.28e-02,  2.83e-02,  2.87e-02,  9.83e-03,  2.62e-02,  -6.45e-03
),
cutseel = cms.vdouble(
1.58e-02,  1.16e-02,  1.15e-02,  3.69e-02,  2.91e-02,  3.26e-02,  1.08e-02,  5.07e-02,  5.43e-02
)
)


eidHZZSuperTight = eidCutBasedExt.clone()
eidHZZSuperTight.electronIDType = 'classbased'
eidHZZSuperTight.electronQuality = 'supertight'
eidHZZSuperTight.electronVersion = 'V06'
eidHZZSuperTight.additionalCategories = True
eidHZZSuperTight.classbasedsupertightEleIDCutsV06 = cms.PSet(
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutip_gsf = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutip_gsfl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdcotdist = cms.vdouble(
1.15e-02,  1.07e-02,  4.01e-03,  2.97e-02,  2.85e-02,  3.10e-02,  9.34e-04,  3.40e-02,  2.82e-07
),
cutdetain = cms.vdouble(
1.09e-02,  4.23e-03,  5.00e-03,  8.57e-03,  4.94e-03,  7.13e-03,  9.39e-03,  1.07e-02,  -9.62e-02
),
cutdetainl = cms.vdouble(
8.35e-03,  3.98e-03,  5.01e-03,  1.51e-02,  5.07e-03,  7.25e-03,  1.27e-02,  1.73e-02,  1.27e-02
),
cutdphiin = cms.vdouble(
5.74e-02,  2.31e-02,  1.37e-01,  6.90e-02,  1.53e-02,  6.50e-02,  3.81e-02,  7.93e-02,  -6.19e-02
),
cutdphiinl = cms.vdouble(
1.76e-02,  2.54e-02,  1.61e-01,  4.90e-02,  1.63e-02,  9.80e-02,  5.50e-02,  9.63e-02,  4.42e-02
),
cuteseedopcor = cms.vdouble(
9.31e-01,  9.48e-01,  8.76e-01,  8.82e-01,  7.80e-01,  7.40e-01,  4.23e-01,  9.43e-01,  6.62e-01
),
cutfmishits = cms.vdouble(
1.50e+00,  1.50e+00,  5.00e-01,  1.50e+00,  1.50e+00,  5.00e-01,  5.00e-01,  5.00e-01,  -5.00e-01
),
cuthoe = cms.vdouble(
8.11e-02,  3.93e-02,  3.83e-02,  1.01e-01,  1.22e-02,  2.33e-02,  1.18e-01,  2.07e-01,  -6.38e-01
),
cuthoel = cms.vdouble(
4.86e-02,  4.25e-02,  3.22e-02,  7.06e-02,  1.45e-02,  2.33e-02,  3.03e-01,  3.80e-01,  8.07e-02
),
cutsee = cms.vdouble(
1.15e-02,  1.06e-02,  1.01e-02,  3.24e-02,  2.79e-02,  2.78e-02,  9.56e-03,  2.60e-02,  -9.74e-02
),
cutseel = cms.vdouble(
1.10e-02,  1.07e-02,  1.02e-02,  2.90e-02,  2.74e-02,  2.77e-02,  9.56e-03,  2.70e-02,  2.75e-02
)
)


eidHZZHyperTight1 = eidCutBasedExt.clone()
eidHZZHyperTight1.electronIDType = 'classbased'
eidHZZHyperTight1.electronQuality = 'hypertight1'
eidHZZHyperTight1.electronVersion = 'V06'
eidHZZHyperTight1.additionalCategories = True
eidHZZHyperTight1.classbasedhypertight1EleIDCutsV06 = cms.PSet(
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutip_gsf = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutip_gsfl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdcotdist = cms.vdouble(
2.70e-04,  1.43e-04,  1.95e-04,  2.64e-03,  2.82e-02,  1.64e-02,  2.61e-05,  2.57e-02,  2.82e-07
),
cutdetain = cms.vdouble(
6.70e-03,  3.57e-03,  3.98e-03,  8.57e-03,  4.86e-03,  6.10e-03,  8.92e-03,  8.96e-03,  -9.62e-02
),
cutdetainl = cms.vdouble(
5.05e-03,  3.68e-03,  3.70e-03,  8.21e-03,  4.86e-03,  6.27e-03,  1.27e-02,  1.50e-02,  1.27e-02
),
cutdphiin = cms.vdouble(
3.06e-02,  1.70e-02,  4.42e-02,  3.01e-02,  1.43e-02,  2.59e-02,  2.31e-02,  4.95e-02,  -6.19e-02
),
cutdphiinl = cms.vdouble(
1.45e-02,  1.56e-02,  4.99e-02,  1.75e-02,  1.36e-02,  3.07e-02,  3.73e-02,  4.42e-02,  4.42e-02
),
cuteseedopcor = cms.vdouble(
1.04e+00,  9.67e-01,  1.04e+00,  9.07e-01,  8.86e-01,  7.67e-01,  7.83e-01,  9.53e-01,  6.62e-01
),
cutfmishits = cms.vdouble(
1.50e+00,  1.50e+00,  5.00e-01,  5.00e-01,  5.00e-01,  5.00e-01,  5.00e-01,  5.00e-01,  -5.00e-01
),
cuthoe = cms.vdouble(
4.68e-02,  3.43e-02,  3.66e-02,  6.69e-02,  9.84e-03,  1.71e-02,  8.41e-02,  4.25e-02,  -6.38e-01
),
cuthoel = cms.vdouble(
3.19e-02,  3.47e-02,  3.19e-02,  4.16e-02,  9.89e-03,  1.98e-02,  2.14e-01,  5.03e-02,  8.07e-02
),
cutsee = cms.vdouble(
1.06e-02,  1.05e-02,  9.84e-03,  3.03e-02,  2.75e-02,  2.74e-02,  9.48e-03,  2.56e-02,  -9.74e-02
),
cutseel = cms.vdouble(
1.00e-02,  1.03e-02,  9.67e-03,  2.75e-02,  2.70e-02,  2.68e-02,  9.41e-03,  2.55e-02,  2.75e-02
)
)

