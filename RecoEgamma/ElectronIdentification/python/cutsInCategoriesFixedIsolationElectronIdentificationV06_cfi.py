import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi import *

eidVeryLoose = eidCutBasedExt.clone()
eidVeryLoose.electronIDType = 'classbased'
eidVeryLoose.electronQuality = 'veryloose'
eidVeryLoose.electronVersion = 'V06'
eidVeryLoose.additionalCategories = True
eidVeryLoose.classbasedverylooseEleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdetain = cms.vdouble(
1.28e-02, 1.50e-02, 2.45e-02, 2.46e-02, 5.51e-02, 4.01e-02, 4.33e-02, 3.08e-02, 3.43e-02
),
cutdetainl = cms.vdouble(
1.25e-02, 1.42e-02, 2.52e-02, 2.54e-02, 6.02e-02, 4.03e-02, 2.24e-02, 1.51e-01, 4.96e-02
),
cutdphiin = cms.vdouble(
8.07e-02, 3.14e-01, 3.57e-01, 1.14e-01, 4.54e-01, 3.77e-01, 4.46e-01, 4.04e-01, 8.54e-01
),
cutdphiinl = cms.vdouble(
7.28e-02, 3.17e-01, 3.57e-01, 9.51e-02, 4.63e-01, 4.33e-01, 3.47e-01, 6.49e-01, 2.97e-01
),
cuteseedopcor = cms.vdouble(
6.32e-01, 2.85e-01, 4.06e-01, 7.36e-01, 3.72e-01, 4.55e-01, 2.49e-01, 3.33e-01, 2.32e-01
),
cutfmishits = cms.vdouble(
4.50e+00, 1.50e+00, 1.50e+00, 8.50e+00, 2.50e+00, 4.50e+00, 3.50e+00, 4.50e+00, 8.50e+00
),
cuthoe = cms.vdouble(
2.16e-01, 1.44e-01, 1.47e-01, 3.71e-01, 1.27e-01, 1.46e-01, 4.55e-01, 4.28e-01, 4.06e-01
),
cuthoel = cms.vdouble(
2.12e-01, 1.38e-01, 1.46e-01, 3.72e-01, 9.22e-02, 1.46e-01, 3.53e-01, 3.81e-01, 3.88e-01
),
cutip_gsf = cms.vdouble(
4.72e-02, 9.36e-02, 1.45e-01, 1.90e-01, 5.97e-01, 6.16e-01, 3.04e-01, 1.45e+00, 2.62e-01
),
cutip_gsfl = cms.vdouble(
4.48e-02, 1.02e-01, 1.45e-01, 1.42e-01, 9.01e-01, 6.26e-01, 1.69e-01, 6.66e-01, 1.32e-01
),
cutsee = cms.vdouble(
1.70e-02, 1.62e-02, 1.89e-02, 5.22e-02, 3.99e-02, 5.10e-02, 1.99e-02, 6.30e-02, 6.01e-02
),
cutseel = cms.vdouble(
1.70e-02, 1.41e-02, 1.89e-02, 5.11e-02, 3.44e-02, 5.10e-02, 2.00e-02, 5.82e-02, 1.16e-01
)
)

eidLoose = eidCutBasedExt.clone()
eidLoose.electronIDType = 'classbased'
eidLoose.electronQuality = 'loose'
eidLoose.electronVersion = 'V06'
eidLoose.additionalCategories = True
eidLoose.classbasedlooseEleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdetain = cms.vdouble(
1.28e-02, 9.17e-03, 2.45e-02, 2.46e-02, 1.69e-02, 2.45e-02, 2.75e-02, 3.07e-02, 2.35e-02
),
cutdetainl = cms.vdouble(
1.25e-02, 8.10e-03, 2.29e-02, 2.54e-02, 1.15e-02, 2.14e-02, 1.98e-02, 1.51e-01, 3.03e-02
),
cutdphiin = cms.vdouble(
8.07e-02, 2.88e-01, 3.52e-01, 1.13e-01, 3.27e-01, 3.41e-01, 3.53e-01, 4.04e-01, 3.42e-01
),
cutdphiinl = cms.vdouble(
7.28e-02, 2.85e-01, 3.53e-01, 9.46e-02, 3.20e-01, 3.36e-01, 3.39e-01, 6.49e-01, 2.93e-01
),
cuteseedopcor = cms.vdouble(
6.33e-01, 3.13e-01, 4.06e-01, 7.36e-01, 4.01e-01, 4.55e-01, 2.50e-01, 3.35e-01, 2.32e-01
),
cutfmishits = cms.vdouble(
4.50e+00, 1.50e+00, 1.50e+00, 8.50e+00, 2.50e+00, 1.50e+00, 3.50e+00, 3.50e+00, 3.50e+00
),
cuthoe = cms.vdouble(
2.16e-01, 1.18e-01, 1.47e-01, 3.71e-01, 6.25e-02, 1.23e-01, 4.54e-01, 4.27e-01, 4.06e-01
),
cuthoel = cms.vdouble(
2.12e-01, 7.55e-02, 1.46e-01, 3.72e-01, 3.48e-02, 1.46e-01, 3.53e-01, 3.81e-01, 3.88e-01
),
cutip_gsf = cms.vdouble(
4.69e-02, 8.45e-02, 1.45e-01, 1.02e-01, 5.90e-01, 3.64e-01, 1.12e-01, 1.31e+00, 1.06e-01
),
cutip_gsfl = cms.vdouble(
4.48e-02, 8.07e-02, 1.45e-01, 6.77e-02, 6.50e-01, 3.13e-01, 1.04e-01, 4.40e-01, 5.41e-02
),
cutsee = cms.vdouble(
1.70e-02, 1.40e-02, 1.89e-02, 5.22e-02, 3.63e-02, 4.93e-02, 1.97e-02, 6.30e-02, 6.01e-02
),
cutseel = cms.vdouble(
1.70e-02, 1.16e-02, 1.89e-02, 5.11e-02, 3.17e-02, 5.06e-02, 2.00e-02, 5.82e-02, 9.96e-02
)
)

eidMedium = eidCutBasedExt.clone()
eidMedium.electronIDType = 'classbased'
eidMedium.electronQuality = 'medium'
eidMedium.electronVersion = 'V06'
eidMedium.additionalCategories = True
eidMedium.classbasedmediumEleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdetain = cms.vdouble(
1.28e-02, 5.54e-03, 1.64e-02, 2.27e-02, 7.92e-03, 1.34e-02, 2.15e-02, 3.07e-02, 1.82e-02
),
cutdetainl = cms.vdouble(
9.08e-03, 4.44e-03, 1.43e-02, 2.53e-02, 6.53e-03, 1.22e-02, 1.98e-02, 1.51e-01, 1.11e-02
),
cutdphiin = cms.vdouble(
7.84e-02, 2.63e-01, 3.04e-01, 1.12e-01, 1.65e-01, 2.80e-01, 3.40e-01, 4.04e-01, 3.26e-01
),
cutdphiinl = cms.vdouble(
7.19e-02, 2.50e-01, 2.91e-01, 9.43e-02, 1.08e-01, 2.78e-01, 3.37e-01, 6.49e-01, 2.47e-01
),
cuteseedopcor = cms.vdouble(
6.35e-01, 8.20e-01, 4.24e-01, 7.36e-01, 6.32e-01, 5.43e-01, 2.50e-01, 8.14e-01, 2.32e-01
),
cutfmishits = cms.vdouble(
2.50e+00, 1.50e+00, 1.50e+00, 2.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 3.50e+00, 1.50e+00
),
cuthoe = cms.vdouble(
2.16e-01, 1.01e-01, 1.47e-01, 3.71e-01, 3.63e-02, 1.22e-01, 4.53e-01, 4.27e-01, 4.06e-01
),
cuthoel = cms.vdouble(
2.12e-01, 7.55e-02, 1.46e-01, 3.72e-01, 2.86e-02, 9.36e-02, 3.53e-01, 3.81e-01, 2.80e-01
),
cutip_gsf = cms.vdouble(
2.13e-02, 7.73e-02, 1.42e-01, 5.81e-02, 2.61e-01, 3.40e-01, 1.12e-01, 1.29e+00, 8.53e-02
),
cutip_gsfl = cms.vdouble(
2.81e-02, 6.63e-02, 1.30e-01, 3.82e-02, 5.30e-01, 1.93e-01, 1.04e-01, 4.40e-01, 3.92e-02
),
cutsee = cms.vdouble(
1.70e-02, 1.17e-02, 1.59e-02, 5.16e-02, 3.62e-02, 3.67e-02, 1.97e-02, 5.12e-02, 4.94e-02
),
cutseel = cms.vdouble(
1.70e-02, 1.10e-02, 1.26e-02, 5.11e-02, 2.85e-02, 3.38e-02, 1.14e-02, 3.60e-02, 4.76e-02
)
)

eidTight = eidCutBasedExt.clone()
eidTight.electronIDType = 'classbased'
eidTight.electronQuality = 'tight'
eidTight.electronVersion = 'V06'
eidTight.additionalCategories = True
eidTight.classbasedtightEleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdetain = cms.vdouble(
1.13e-02, 4.38e-03, 1.29e-02, 1.34e-02, 7.30e-03, 1.14e-02, 2.15e-02, 3.07e-02, 1.20e-02
),
cutdetainl = cms.vdouble(
8.50e-03, 4.14e-03, 1.43e-02, 2.26e-02, 6.38e-03, 9.04e-03, 1.98e-02, 1.51e-01, 6.04e-03
),
cutdphiin = cms.vdouble(
7.43e-02, 2.30e-01, 2.64e-01, 1.02e-01, 6.14e-02, 2.52e-01, 3.40e-01, 4.04e-01, 3.26e-01
),
cutdphiinl = cms.vdouble(
6.90e-02, 1.61e-01, 2.91e-01, 9.17e-02, 4.38e-02, 2.46e-01, 3.37e-01, 6.49e-01, 2.47e-01
),
cuteseedopcor = cms.vdouble(
6.35e-01, 9.35e-01, 7.90e-01, 7.39e-01, 8.32e-01, 6.18e-01, 2.56e-01, 8.14e-01, 6.11e-01
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 5.00e-01, 2.50e+00, 1.50e+00, 5.00e-01, 1.50e+00, 2.50e+00, 5.00e-01
),
cuthoe = cms.vdouble(
2.16e-01, 8.72e-02, 1.47e-01, 3.70e-01, 3.34e-02, 1.08e-01, 4.53e-01, 4.22e-01, 3.98e-01
),
cuthoel = cms.vdouble(
2.12e-01, 6.56e-02, 1.44e-01, 3.71e-01, 2.86e-02, 9.36e-02, 3.53e-01, 3.81e-01, 2.72e-01
),
cutip_gsf = cms.vdouble(
1.88e-02, 6.96e-02, 1.29e-01, 3.71e-02, 2.01e-01, 2.81e-01, 1.12e-01, 2.23e-01, 5.35e-02
),
cutip_gsfl = cms.vdouble(
1.20e-02, 6.56e-02, 1.15e-01, 2.05e-02, 2.26e-01, 1.93e-01, 1.04e-01, 1.35e-01, 3.50e-02
),
cutsee = cms.vdouble(
1.55e-02, 1.17e-02, 1.25e-02, 5.16e-02, 3.03e-02, 3.03e-02, 1.07e-02, 3.72e-02, 4.94e-02
),
cutseel = cms.vdouble(
1.70e-02, 1.10e-02, 1.07e-02, 4.85e-02, 2.80e-02, 2.96e-02, 9.62e-03, 2.85e-02, 4.76e-02
)
)

eidSuperTight = eidCutBasedExt.clone()
eidSuperTight.electronIDType = 'classbased'
eidSuperTight.electronQuality = 'supertight'
eidSuperTight.electronVersion = 'V06'
eidSuperTight.additionalCategories = True
eidSuperTight.classbasedsupertightEleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdetain = cms.vdouble(
1.08e-02, 3.63e-03, 7.42e-03, 1.34e-02, 5.75e-03, 8.64e-03, 1.07e-02, 2.70e-02, 3.59e-03
),
cutdetainl = cms.vdouble(
7.99e-03, 3.41e-03, 6.59e-03, 1.34e-02, 5.66e-03, 9.04e-03, 1.43e-02, 4.72e-02, 4.35e-03
),
cutdphiin = cms.vdouble(
7.36e-02, 8.18e-02, 1.98e-01, 1.02e-01, 3.66e-02, 2.46e-01, 9.45e-02, 2.12e-01, 3.26e-01
),
cutdphiinl = cms.vdouble(
6.87e-02, 3.76e-02, 2.91e-01, 6.66e-02, 2.02e-02, 2.25e-01, 1.53e-01, 3.73e-01, 2.47e-01
),
cuteseedopcor = cms.vdouble(
6.45e-01, 9.49e-01, 8.86e-01, 7.66e-01, 8.37e-01, 9.76e-01, 4.30e-01, 8.48e-01, 6.11e-01
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 5.00e-01, 2.50e+00, 1.50e+00, 5.00e-01, 1.50e+00, 2.50e+00, -5.00e-01
),
cuthoe = cms.vdouble(
1.51e-01, 7.01e-02, 1.47e-01, 3.69e-01, 3.34e-02, 9.19e-02, 4.53e-01, 4.22e-01, 3.98e-01
),
cuthoel = cms.vdouble(
3.58e-02, 5.04e-02, 1.44e-01, 3.71e-01, 2.09e-02, 8.45e-02, 2.97e-01, 3.81e-01, 2.72e-01
),
cutip_gsf = cms.vdouble(
1.88e-02, 4.84e-02, 1.09e-01, 1.76e-02, 1.79e-01, 2.57e-01, 1.03e-01, 1.64e-01, 4.63e-02
),
cutip_gsfl = cms.vdouble(
1.06e-02, 4.45e-02, 1.07e-01, 1.04e-02, 1.80e-01, 1.33e-01, 9.95e-02, 1.35e-01, 3.22e-02
),
cutsee = cms.vdouble(
1.41e-02, 1.10e-02, 1.25e-02, 3.88e-02, 2.99e-02, 2.90e-02, 1.00e-02, 3.72e-02, 4.55e-02
),
cutseel = cms.vdouble(
1.39e-02, 1.09e-02, 1.00e-02, 3.35e-02, 2.78e-02, 2.74e-02, 9.42e-03, 2.59e-02, 3.62e-02
)
)

eidHyperTight1 = eidCutBasedExt.clone()
eidHyperTight1.electronIDType = 'classbased'
eidHyperTight1.electronQuality = 'hypertight1'
eidHyperTight1.electronVersion = 'V06'
eidHyperTight1.additionalCategories = True
eidHyperTight1.classbasedhypertight1EleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdetain = cms.vdouble(
1.08e-02, 3.63e-03, 4.15e-03, 1.34e-02, 4.44e-03, 6.61e-03, 8.89e-03, 1.04e-02, 3.59e-03
),
cutdetainl = cms.vdouble(
4.03e-03, 3.11e-03, 4.09e-03, 1.34e-02, 3.39e-03, 7.24e-03, 7.63e-03, 4.72e-02, 4.35e-03
),
cutdphiin = cms.vdouble(
5.64e-02, 3.34e-02, 1.98e-01, 5.41e-02, 2.90e-02, 9.35e-02, 7.89e-02, 2.12e-01, 3.26e-01
),
cutdphiinl = cms.vdouble(
5.72e-02, 1.94e-02, 2.91e-01, 4.55e-02, 1.50e-02, 8.71e-02, 1.25e-01, 3.73e-01, 2.47e-01
),
cuteseedopcor = cms.vdouble(
7.18e-01, 9.58e-01, 1.01e+00, 8.44e-01, 8.37e-01, 9.98e-01, 4.30e-01, 9.64e-01, 6.11e-01
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 5.00e-01, 5.00e-01, 1.50e+00, 5.00e-01, 1.50e+00, 1.50e+00, -5.00e-01
),
cuthoe = cms.vdouble(
8.99e-02, 7.01e-02, 1.47e-01, 1.67e-01, 2.38e-02, 7.58e-02, 4.53e-01, 2.85e-01, 3.98e-01
),
cuthoel = cms.vdouble(
3.58e-02, 4.54e-02, 1.21e-01, 1.81e-01, 1.83e-02, 6.84e-02, 1.76e-01, 3.69e-01, 2.72e-01
),
cutip_gsf = cms.vdouble(
1.36e-02, 1.91e-02, 8.11e-02, 1.70e-02, 1.22e-01, 6.18e-02, 8.18e-02, 1.39e-01, 4.63e-02
),
cutip_gsfl = cms.vdouble(
1.06e-02, 2.29e-02, 7.57e-02, 8.60e-03, 1.18e-01, 1.33e-01, 8.17e-02, 1.18e-01, 3.22e-02
),
cutsee = cms.vdouble(
1.12e-02, 1.08e-02, 1.12e-02, 3.88e-02, 2.99e-02, 2.79e-02, 9.69e-03, 3.12e-02, 4.55e-02
),
cutseel = cms.vdouble(
1.39e-02, 1.02e-02, 9.78e-03, 2.95e-02, 2.71e-02, 2.71e-02, 9.34e-03, 2.56e-02, 3.62e-02
)
)

eidHyperTight2 = eidCutBasedExt.clone()
eidHyperTight2.electronIDType = 'classbased'
eidHyperTight2.electronQuality = 'hypertight2'
eidHyperTight2.electronVersion = 'V06'
eidHyperTight2.additionalCategories = True
eidHyperTight2.classbasedhypertight2EleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdetain = cms.vdouble(
6.26e-03, 3.13e-03, 3.47e-03, 1.34e-02, 3.98e-03, 6.13e-03, 8.89e-03, 9.68e-03, 3.59e-03
),
cutdetainl = cms.vdouble(
3.14e-03, 2.08e-03, 3.07e-03, 1.34e-02, 3.39e-03, 6.04e-03, 7.63e-03, 1.19e-02, 4.35e-03
),
cutdphiin = cms.vdouble(
5.64e-02, 3.34e-02, 1.98e-01, 2.21e-02, 2.90e-02, 9.35e-02, 5.54e-02, 1.20e-01, 3.26e-01
),
cutdphiinl = cms.vdouble(
3.95e-02, 1.94e-02, 2.62e-01, 1.62e-02, 1.50e-02, 8.71e-02, 4.93e-02, 2.45e-01, 2.47e-01
),
cuteseedopcor = cms.vdouble(
7.30e-01, 9.58e-01, 1.01e+00, 8.90e-01, 8.37e-01, 9.98e-01, 4.30e-01, 9.64e-01, 6.11e-01
),
cutfmishits = cms.vdouble(
5.00e-01, 1.50e+00, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 1.50e+00, 5.00e-01, -5.00e-01
),
cuthoe = cms.vdouble(
6.94e-02, 5.48e-02, 5.46e-02, 8.59e-02, 1.59e-02, 5.51e-02, 1.15e-01, 2.85e-01, 3.98e-01
),
cuthoel = cms.vdouble(
3.58e-02, 3.81e-02, 4.87e-02, 8.30e-02, 1.45e-02, 4.62e-02, 8.30e-02, 3.69e-01, 2.72e-01
),
cutip_gsf = cms.vdouble(
1.12e-02, 1.27e-02, 7.52e-02, 1.18e-02, 1.16e-01, 6.18e-02, 7.83e-02, 1.32e-01, 4.63e-02
),
cutip_gsfl = cms.vdouble(
8.30e-03, 6.51e-03, 7.41e-02, 8.60e-03, 1.18e-01, 7.65e-02, 5.37e-02, 1.05e-01, 3.22e-02
),
cutsee = cms.vdouble(
1.11e-02, 1.08e-02, 1.03e-02, 2.94e-02, 2.92e-02, 2.73e-02, 9.53e-03, 2.91e-02, 4.55e-02
),
cutseel = cms.vdouble(
1.33e-02, 1.00e-02, 9.40e-03, 2.79e-02, 2.71e-02, 2.55e-02, 9.34e-03, 2.55e-02, 3.62e-02
)
)


eidHyperTight3 = eidCutBasedExt.clone()
eidHyperTight3.electronIDType = 'classbased'
eidHyperTight3.electronQuality = 'hypertight3'
eidHyperTight3.electronVersion = 'V06'
eidHyperTight3.additionalCategories = True
eidHyperTight3.classbasedhypertight3EleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdetain = cms.vdouble(
6.26e-03, 3.13e-03, 3.47e-03, 1.34e-02, 3.98e-03, 6.13e-03, 8.89e-03, 9.68e-03, 3.59e-03
),
cutdetainl = cms.vdouble(
3.14e-03, 2.08e-03, 3.07e-03, 1.34e-02, 3.39e-03, 6.04e-03, 7.63e-03, 1.19e-02, 4.35e-03
),
cutdphiin = cms.vdouble(
5.64e-02, 3.34e-02, 1.98e-01, 2.21e-02, 2.90e-02, 9.35e-02, 5.54e-02, 1.20e-01, 3.26e-01
),
cutdphiinl = cms.vdouble(
3.95e-02, 1.94e-02, 2.62e-01, 1.62e-02, 1.50e-02, 8.71e-02, 4.93e-02, 2.45e-01, 2.47e-01
),
cuteseedopcor = cms.vdouble(
7.30e-01, 9.58e-01, 1.01e+00, 8.90e-01, 8.37e-01, 9.98e-01, 4.30e-01, 9.64e-01, 6.11e-01
),
cutfmishits = cms.vdouble(
5.00e-01, 1.50e+00, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 1.50e+00, 5.00e-01, -5.00e-01
),
cuthoe = cms.vdouble(
6.94e-02, 5.48e-02, 5.46e-02, 8.59e-02, 1.59e-02, 5.51e-02, 1.15e-01, 2.85e-01, 3.98e-01
),
cuthoel = cms.vdouble(
3.58e-02, 3.81e-02, 4.87e-02, 8.30e-02, 1.45e-02, 4.62e-02, 8.30e-02, 3.69e-01, 2.72e-01
),
cutip_gsf = cms.vdouble(
1.12e-02, 1.27e-02, 7.52e-02, 1.18e-02, 1.16e-01, 6.18e-02, 7.83e-02, 1.32e-01, 4.63e-02
),
cutip_gsfl = cms.vdouble(
8.30e-03, 6.51e-03, 7.41e-02, 8.60e-03, 1.18e-01, 7.65e-02, 5.37e-02, 1.05e-01, 3.22e-02
),
cutsee = cms.vdouble(
1.11e-02, 1.08e-02, 1.03e-02, 2.94e-02, 2.92e-02, 2.73e-02, 9.53e-03, 2.91e-02, 4.55e-02
),
cutseel = cms.vdouble(
1.33e-02, 1.00e-02, 9.40e-03, 2.79e-02, 2.71e-02, 2.55e-02, 9.34e-03, 2.55e-02, 3.62e-02
)
)

eidHyperTight4 = eidCutBasedExt.clone()
eidHyperTight4.electronIDType = 'classbased'
eidHyperTight4.electronQuality = 'hypertight4'
eidHyperTight4.electronVersion = 'V06'
eidHyperTight4.additionalCategories = True
eidHyperTight4.classbasedhypertight4EleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdetain = cms.vdouble(
6.26e-03, 3.13e-03, 3.47e-03, 1.34e-02, 3.98e-03, 6.13e-03, 8.89e-03, 9.68e-03, 3.59e-03
),
cutdetainl = cms.vdouble(
3.14e-03, 2.08e-03, 3.07e-03, 1.34e-02, 3.39e-03, 6.04e-03, 7.63e-03, 1.19e-02, 4.35e-03
),
cutdphiin = cms.vdouble(
5.64e-02, 3.34e-02, 1.98e-01, 2.21e-02, 2.90e-02, 9.35e-02, 5.54e-02, 1.20e-01, 3.26e-01
),
cutdphiinl = cms.vdouble(
3.95e-02, 1.94e-02, 2.62e-01, 1.62e-02, 1.50e-02, 8.71e-02, 4.93e-02, 2.45e-01, 2.47e-01
),
cuteseedopcor = cms.vdouble(
7.30e-01, 9.58e-01, 1.01e+00, 8.90e-01, 8.37e-01, 9.98e-01, 4.30e-01, 9.64e-01, 6.11e-01
),
cutfmishits = cms.vdouble(
5.00e-01, 1.50e+00, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 1.50e+00, 5.00e-01, -5.00e-01
),
cuthoe = cms.vdouble(
6.94e-02, 5.48e-02, 5.46e-02, 8.59e-02, 1.59e-02, 5.51e-02, 1.15e-01, 2.85e-01, 3.98e-01
),
cuthoel = cms.vdouble(
3.58e-02, 3.81e-02, 4.87e-02, 8.30e-02, 1.45e-02, 4.62e-02, 8.30e-02, 3.69e-01, 2.72e-01
),
cutip_gsf = cms.vdouble(
1.12e-02, 1.27e-02, 7.52e-02, 1.18e-02, 1.16e-01, 6.18e-02, 7.83e-02, 1.32e-01, 4.63e-02
),
cutip_gsfl = cms.vdouble(
8.30e-03, 6.51e-03, 7.41e-02, 8.60e-03, 1.18e-01, 7.65e-02, 5.37e-02, 1.05e-01, 3.22e-02
),
cutsee = cms.vdouble(
1.11e-02, 1.08e-02, 1.03e-02, 2.94e-02, 2.92e-02, 2.73e-02, 9.53e-03, 2.91e-02, 4.55e-02
),
cutseel = cms.vdouble(
1.33e-02, 1.00e-02, 9.40e-03, 2.79e-02, 2.71e-02, 2.55e-02, 9.34e-03, 2.55e-02, 3.62e-02
)
)
