import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi import *

eidVeryLoose = eidCutBasedExt.clone()
eidVeryLoose.electronIDType = 'classbased'
eidVeryLoose.electronQuality = 'veryloose'
eidVeryLoose.electronVersion = 'V06'
eidVeryLoose.additionalCategories = True
eidVeryLoose.classbasedverylooseEleIDCutsV06 = cms.PSet(
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdcotdist = cms.vdouble(
3.89e-02, 3.90e-02, 3.96e-02, 3.92e-02, 3.95e-02, 3.97e-02, 3.93e-02, 3.96e-02, 2.66e-02
),
cutdetain = cms.vdouble(
1.29e-02, 8.30e-03, 2.28e-02, 2.53e-02, 5.32e-02, 1.73e-02, 5.24e-02, 3.55e-02, 2.75e-02
),
cutdetainl = cms.vdouble(
1.25e-02, 4.01e-03, 2.78e-02, 2.61e-02, 7.45e-02, 2.99e-02, 1.07e-02, 2.24e-01, 2.84e-02
),
cutdphiin = cms.vdouble(
8.37e-02, 2.80e-01, 3.46e-01, 9.16e-02, 4.05e-01, 2.92e-01, 4.21e-01, 3.99e-01, 2.43e+00
),
cutdphiinl = cms.vdouble(
7.53e-02, 2.88e-01, 3.42e-01, 8.83e-02, 4.27e-01, 3.84e-01, 2.52e-01, 6.92e-01, 1.88e-01
),
cuteseedopcor = cms.vdouble(
6.49e-01, 3.23e-01, 4.12e-01, 7.36e-01, 2.12e-01, 4.67e-01, 3.70e-01, 2.63e-01, 2.11e-01
),
cutfmishits = cms.vdouble(
4.50e+00, 1.50e+00, 1.50e+00, 8.50e+00, 2.50e+00, 4.50e+00, 3.50e+00, 4.50e+00, 8.50e+00
),
cuthoe = cms.vdouble(
2.08e-01, 1.39e-01, 1.48e-01, 3.67e-01, 5.85e-02, 1.46e-01, 4.34e-01, 6.07e-01, 4.03e-01
),
cuthoel = cms.vdouble(
2.11e-01, 1.29e-01, 1.48e-01, 3.59e-01, 3.52e-02, 1.46e-01, 3.17e-01, 3.43e-01, 3.88e-01
),
cutip_gsf = cms.vdouble(
7.36e-02, 1.02e-01, 1.77e-01, 2.63e-01, 2.59e-01, 3.01e-01, 4.83e-01, 1.51e+00, 3.67e-01
),
cutip_gsfl = cms.vdouble(
9.40e-02, 1.40e-01, 1.90e-01, 1.44e-01, 8.56e-01, 7.51e-01, 4.86e-01, 6.46e+00, 1.15e-01
),
cutsee = cms.vdouble(
1.69e-02, 1.28e-02, 1.89e-02, 4.91e-02, 2.97e-02, 3.48e-02, 1.78e-02, 6.37e-02, 6.71e-02
),
cutseel = cms.vdouble(
1.69e-02, 1.39e-02, 1.84e-02, 5.01e-02, 4.21e-02, 4.89e-02, 1.91e-02, 5.70e-02, 1.22e-01
)
)

eidLoose = eidCutBasedExt.clone()
eidLoose.electronIDType = 'classbased'
eidLoose.electronQuality = 'loose'
eidLoose.electronVersion = 'V06'
eidLoose.additionalCategories = True
eidLoose.classbasedlooseEleIDCutsV06 = cms.PSet(
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdcotdist = cms.vdouble(
3.89e-02, 3.44e-02, 3.05e-02, 3.92e-02, 3.30e-02, 3.97e-02, 3.10e-02, 3.96e-02, 4.22e-03
),
cutdetain = cms.vdouble(
1.29e-02, 5.30e-03, 1.50e-02, 2.40e-02, 8.14e-03, 1.02e-02, 5.24e-02, 3.54e-02, 1.84e-02
),
cutdetainl = cms.vdouble(
1.25e-02, 2.93e-03, 2.78e-02, 2.61e-02, 3.38e-02, 2.30e-02, 1.07e-02, 2.24e-01, 2.38e-02
),
cutdphiin = cms.vdouble(
8.37e-02, 2.46e-01, 3.28e-01, 8.95e-02, 2.99e-01, 2.52e-01, 3.82e-01, 3.99e-01, 1.78e+00
),
cutdphiinl = cms.vdouble(
7.53e-02, 2.44e-01, 3.08e-01, 8.62e-02, 2.96e-01, 3.18e-01, 2.11e-01, 6.92e-01, 1.28e-01
),
cuteseedopcor = cms.vdouble(
6.49e-01, 8.66e-01, 4.12e-01, 7.36e-01, 2.36e-01, 4.68e-01, 3.70e-01, 2.63e-01, 2.11e-01
),
cutfmishits = cms.vdouble(
4.50e+00, 1.50e+00, 1.50e+00, 8.50e+00, 2.50e+00, 1.50e+00, 3.50e+00, 3.50e+00, 3.50e+00
),
cuthoe = cms.vdouble(
2.08e-01, 7.30e-02, 1.48e-01, 3.67e-01, 2.50e-02, 1.29e-01, 4.34e-01, 6.07e-01, 4.03e-01
),
cuthoel = cms.vdouble(
2.11e-01, 1.01e-01, 1.48e-01, 3.59e-01, 5.01e-04, 1.46e-01, 3.17e-01, 3.43e-01, 3.88e-01
),
cutip_gsf = cms.vdouble(
7.36e-02, 9.41e-02, 1.74e-01, 7.28e-02, 1.68e-01, 1.59e-01, 4.83e-01, 1.22e+00, 7.80e-02
),
cutip_gsfl = cms.vdouble(
9.39e-02, 1.31e-01, 1.89e-01, 4.41e-02, 8.18e-01, 6.54e-01, 4.86e-01, 6.46e+00, 2.61e-02
),
cutsee = cms.vdouble(
1.69e-02, 1.13e-02, 1.32e-02, 4.91e-02, 2.81e-02, 2.98e-02, 1.10e-02, 4.67e-02, 5.11e-02
),
cutseel = cms.vdouble(
1.69e-02, 1.28e-02, 1.78e-02, 5.01e-02, 4.05e-02, 4.61e-02, 1.63e-02, 5.60e-02, 1.09e-01
)
)

eidMedium = eidCutBasedExt.clone()
eidMedium.electronIDType = 'classbased'
eidMedium.electronQuality = 'medium'
eidMedium.electronVersion = 'V06'
eidMedium.additionalCategories = True
eidMedium.classbasedmediumEleIDCutsV06 = cms.PSet(
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdcotdist = cms.vdouble(
3.31e-02, 2.70e-02, 2.01e-02, 3.92e-02, 2.74e-02, 2.37e-02, 2.52e-02, 3.48e-02, 1.18e-04
),
cutdetain = cms.vdouble(
1.29e-02, 4.05e-03, 7.77e-03, 1.21e-02, 6.45e-03, 7.15e-03, 1.28e-02, 3.54e-02, 1.26e-02
),
cutdetainl = cms.vdouble(
1.25e-02, 2.72e-03, 2.30e-02, 2.08e-02, 3.38e-02, 2.06e-02, 1.03e-02, 2.24e-01, 2.38e-02
),
cutdphiin = cms.vdouble(
8.37e-02, 2.37e-01, 2.96e-01, 8.40e-02, 4.46e-02, 1.18e-01, 2.06e-01, 3.99e-01, 1.78e+00
),
cutdphiinl = cms.vdouble(
7.51e-02, 2.44e-01, 2.38e-01, 8.31e-02, 8.79e-02, 1.40e-01, 1.71e-01, 6.92e-01, 1.28e-01
),
cuteseedopcor = cms.vdouble(
6.49e-01, 9.44e-01, 8.45e-01, 7.36e-01, 4.67e-01, 4.79e-01, 3.70e-01, 9.09e-01, 5.85e-01
),
cutfmishits = cms.vdouble(
2.50e+00, 1.50e+00, 1.50e+00, 2.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 3.50e+00, 1.50e+00
),
cuthoe = cms.vdouble(
2.08e-01, 4.38e-02, 1.47e-01, 3.63e-01, 7.62e-03, 5.32e-02, 4.34e-01, 6.07e-01, 3.38e-01
),
cuthoel = cms.vdouble(
2.11e-01, 8.73e-02, 1.48e-01, 3.48e-01, -1.35e-02, 8.45e-02, 3.17e-01, 3.43e-01, 3.29e-01
),
cutip_gsf = cms.vdouble(
7.36e-02, 9.25e-02, 8.11e-02, 3.97e-02, 1.12e-01, 9.73e-02, 4.83e-01, 1.14e+00, 1.89e-02
),
cutip_gsfl = cms.vdouble(
9.39e-02, 1.29e-01, 1.36e-01, 2.06e-02, 8.18e-01, 6.38e-01, 2.59e-01, 6.46e+00, 2.61e-02
),
cutsee = cms.vdouble(
1.42e-02, 1.08e-02, 1.07e-02, 3.54e-02, 2.81e-02, 2.80e-02, 9.93e-03, 3.06e-02, 2.81e-02
),
cutseel = cms.vdouble(
1.69e-02, 1.24e-02, 1.62e-02, 4.69e-02, 4.05e-02, 4.46e-02, 1.57e-02, 4.64e-02, 8.90e-02
)
)

eidTight = eidCutBasedExt.clone()
eidTight.electronIDType = 'classbased'
eidTight.electronQuality = 'tight'
eidTight.electronVersion = 'V06'
eidTight.additionalCategories = True
eidTight.classbasedtightEleIDCutsV06 = cms.PSet(
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdcotdist = cms.vdouble(
2.42e-02, 1.59e-02, 9.57e-03, 3.42e-02, 1.95e-02, 8.55e-03, 1.99e-02, 2.35e-02, 9.18e-06
),
cutdetain = cms.vdouble(
9.15e-03, 3.32e-03, 5.79e-03, 1.07e-02, 5.49e-03, 6.26e-03, 1.11e-02, 2.41e-02, -4.46e-04
),
cutdetainl = cms.vdouble(
8.78e-03, 2.28e-03, 2.25e-02, 1.90e-02, 3.38e-02, 1.97e-02, 1.03e-02, 2.10e-01, 5.06e-03
),
cutdphiin = cms.vdouble(
7.78e-02, 5.58e-02, 7.93e-02, 6.81e-02, 1.89e-02, 2.38e-02, 6.09e-02, 1.96e-01, 3.30e-01
),
cutdphiinl = cms.vdouble(
7.24e-02, 1.12e-01, 9.19e-02, 8.26e-02, 7.66e-02, 4.63e-02, 6.10e-02, 6.92e-01, 8.35e-02
),
cuteseedopcor = cms.vdouble(
6.65e-01, 9.75e-01, 9.78e-01, 7.63e-01, 4.67e-01, 9.13e-01, 8.84e-01, 1.00e+00, 6.11e-01
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 5.00e-01, 2.50e+00, 1.50e+00, 5.00e-01, 1.50e+00, 2.50e+00, 5.00e-01
),
cuthoe = cms.vdouble(
1.34e-01, 3.38e-02, 9.48e-02, 3.63e-01, 6.04e-03, 2.90e-02, 4.26e-01, 6.07e-01, -3.64e-02
),
cuthoel = cms.vdouble(
1.12e-01, 8.18e-02, 7.96e-02, 3.48e-01, -1.35e-02, 7.19e-02, 3.17e-01, 3.39e-01, 3.29e-01
),
cutip_gsf = cms.vdouble(
1.98e-02, 6.55e-02, 4.82e-02, 1.98e-02, 6.48e-02, 5.24e-02, 5.08e-02, 1.60e-01, -5.13e-03
),
cutip_gsfl = cms.vdouble(
4.04e-02, 1.21e-01, 1.26e-01, 1.03e-02, 8.18e-01, 6.35e-01, 1.61e-02, 5.51e+00, 2.61e-02
),
cutsee = cms.vdouble(
1.14e-02, 1.06e-02, 1.03e-02, 3.19e-02, 2.77e-02, 2.76e-02, 9.86e-03, 2.60e-02, -2.14e-02
),
cutseel = cms.vdouble(
1.44e-02, 1.23e-02, 1.60e-02, 4.51e-02, 4.05e-02, 4.46e-02, 1.57e-02, 4.35e-02, 5.64e-02
)
)

eidSuperTight = eidCutBasedExt.clone()
eidSuperTight.electronIDType = 'classbased'
eidSuperTight.electronQuality = 'supertight'
eidSuperTight.electronVersion = 'V06'
eidSuperTight.additionalCategories = True
eidSuperTight.classbasedsupertightEleIDCutsV06 = cms.PSet(
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdcotdist = cms.vdouble(
1.58e-02, 1.18e-02, 2.56e-03, 2.76e-02, 6.01e-03, 2.39e-04, 1.46e-02, 9.49e-03, 9.18e-06
),
cutdetain = cms.vdouble(
7.33e-03, 2.68e-03, 5.57e-03, 8.15e-03, 4.42e-03, 5.00e-03, 1.02e-02, 1.63e-02, -4.46e-04
),
cutdetainl = cms.vdouble(
5.88e-03, 1.44e-03, 2.25e-02, 1.56e-02, 3.38e-02, 1.88e-02, 1.01e-02, 2.01e-01, 5.06e-03
),
cutdphiin = cms.vdouble(
5.52e-02, 3.13e-02, 3.78e-02, 3.38e-02, 1.10e-02, 1.18e-02, 3.09e-02, 6.07e-02, 3.30e-01
),
cutdphiinl = cms.vdouble(
5.75e-02, 1.07e-01, 6.23e-02, 8.18e-02, 7.66e-02, 3.45e-02, 3.96e-02, 6.10e-01, 8.35e-02
),
cuteseedopcor = cms.vdouble(
7.30e-01, 9.84e-01, 1.02e+00, 9.63e-01, 9.05e-01, 1.21e+00, 9.85e-01, 1.12e+00, 6.11e-01
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 5.00e-01, 2.50e+00, 1.50e+00, 5.00e-01, 1.50e+00, 2.50e+00, -5.00e-01
),
cuthoe = cms.vdouble(
6.54e-02, 2.96e-02, 4.54e-02, 2.54e-01, 6.04e-03, 2.14e-02, 9.39e-02, 5.11e-01, -3.64e-02
),
cuthoel = cms.vdouble(
4.65e-02, 8.02e-02, 3.41e-02, 3.09e-01, -1.35e-02, 7.00e-02, 1.32e-01, 3.04e-01, 3.29e-01
),
cutip_gsf = cms.vdouble(
1.37e-02, 2.70e-02, 3.15e-02, 1.35e-02, 3.80e-02, 4.38e-02, 2.38e-02, 1.13e-01, -5.13e-03
),
cutip_gsfl = cms.vdouble(
2.98e-02, 1.12e-01, 1.26e-01, 7.76e-03, 8.18e-01, 6.35e-01, 6.66e-03, 5.51e+00, 2.61e-02
),
cutsee = cms.vdouble(
1.07e-02, 1.03e-02, 1.01e-02, 2.99e-02, 2.76e-02, 2.75e-02, 9.86e-03, 2.48e-02, -2.14e-02
),
cutseel = cms.vdouble(
1.44e-02, 1.22e-02, 1.60e-02, 4.26e-02, 4.05e-02, 4.46e-02, 1.57e-02, 4.34e-02, 5.64e-02
)
)

eidHyperTight1 = eidCutBasedExt.clone()
eidHyperTight1.electronIDType = 'classbased'
eidHyperTight1.electronQuality = 'hypertight1'
eidHyperTight1.electronVersion = 'V06'
eidHyperTight1.additionalCategories = True
eidHyperTight1.classbasedhypertight1EleIDCutsV06 = cms.PSet(
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdcotdist = cms.vdouble(
7.87e-03, 2.23e-03, 7.17e-05, 1.15e-02, 4.67e-04, 1.86e-05, 9.28e-03, 3.06e-03, 9.18e-06
),
cutdetain = cms.vdouble(
4.53e-03, 2.10e-03, 4.66e-03, 6.67e-03, 3.48e-03, 3.92e-03, 1.02e-02, 1.48e-02, -4.46e-04
),
cutdetainl = cms.vdouble(
1.83e-03, 3.70e-04, 2.25e-02, 1.42e-02, 3.38e-02, 1.85e-02, 1.01e-02, 1.98e-01, 5.06e-03
),
cutdphiin = cms.vdouble(
2.41e-02, 2.01e-02, 2.14e-02, 2.71e-02, 5.56e-03, 7.07e-03, 1.83e-02, 4.15e-02, 3.30e-01
),
cutdphiinl = cms.vdouble(
2.24e-02, 1.06e-01, 4.66e-02, 7.44e-02, 7.66e-02, 2.96e-02, 3.96e-02, 6.07e-01, 8.35e-02
),
cuteseedopcor = cms.vdouble(
1.03e+00, 1.00e+00, 1.12e+00, 1.03e+00, 9.56e-01, 1.21e+00, 1.02e+00, 1.23e+00, 6.11e-01
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 5.00e-01, 5.00e-01, 1.50e+00, 5.00e-01, 1.50e+00, 1.50e+00, -5.00e-01
),
cuthoe = cms.vdouble(
4.80e-02, 2.81e-02, 3.04e-02, 6.53e-02, 6.04e-03, 1.29e-02, 7.62e-02, 5.75e-02, -3.64e-02
),
cuthoel = cms.vdouble(
-3.89e-03, 7.54e-02, -1.99e-02, 1.18e-01, -1.35e-02, 6.30e-02, 1.24e-01, -1.51e-01, 3.29e-01
),
cutip_gsf = cms.vdouble(
1.14e-02, 1.40e-02, 2.88e-02, 1.05e-02, 2.82e-02, 2.77e-02, 1.49e-02, 1.13e-01, -5.13e-03
),
cutip_gsfl = cms.vdouble(
2.63e-02, 9.82e-02, 1.19e-01, 2.11e-03, 8.18e-01, 6.35e-01, -2.05e-03, 5.51e+00, 2.61e-02
),
cutsee = cms.vdouble(
1.04e-02, 1.02e-02, 9.85e-03, 2.88e-02, 2.67e-02, 2.65e-02, 9.83e-03, 2.42e-02, -2.14e-02
),
cutseel = cms.vdouble(
1.44e-02, 1.22e-02, 1.59e-02, 4.12e-02, 4.05e-02, 4.44e-02, 1.56e-02, 4.34e-02, 5.64e-02
)
)

eidHyperTight2 = eidCutBasedExt.clone()
eidHyperTight2.electronIDType = 'classbased'
eidHyperTight2.electronQuality = 'hypertight2'
eidHyperTight2.electronVersion = 'V06'
eidHyperTight2.additionalCategories = True
eidHyperTight2.classbasedhypertight2EleIDCutsV06 = cms.PSet(
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdcotdist = cms.vdouble(
7.87e-03, 6.25e-05, 7.17e-05, 3.21e-04, 4.67e-04, 1.86e-05, 2.60e-04, 2.38e-04, 9.18e-06
),
cutdetain = cms.vdouble(
4.53e-03, 1.31e-03, 4.62e-03, 5.49e-03, 3.48e-03, 3.92e-03, 9.39e-03, 1.48e-02, -4.46e-04
),
cutdetainl = cms.vdouble(
1.83e-03, -3.04e-04, 2.25e-02, 1.39e-02, 3.38e-02, 1.85e-02, 1.40e-03, 1.96e-01, 5.06e-03
),
cutdphiin = cms.vdouble(
2.00e-02, 1.62e-02, 2.14e-02, 2.06e-02, 5.56e-03, 7.07e-03, 1.47e-02, 2.30e-02, 3.30e-01
),
cutdphiinl = cms.vdouble(
1.49e-02, 1.06e-01, 4.66e-02, 7.44e-02, 7.66e-02, 2.96e-02, 3.96e-02, 6.07e-01, 8.35e-02
),
cuteseedopcor = cms.vdouble(
1.04e+00, 1.01e+00, 1.12e+00, 1.09e+00, 9.56e-01, 1.21e+00, 1.05e+00, 1.34e+00, 6.11e-01
),
cutfmishits = cms.vdouble(
5.00e-01, 1.50e+00, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 1.50e+00, 5.00e-01, -5.00e-01
),
cuthoe = cms.vdouble(
1.50e-02, 2.37e-02, 7.69e-03, 3.71e-02, 6.04e-03, 1.29e-02, 5.86e-02, 4.88e-02, -3.64e-02
),
cuthoel = cms.vdouble(
-3.87e-02, 6.33e-02, -4.16e-02, 9.16e-02, -1.35e-02, 6.30e-02, 1.24e-01, -1.51e-01, 3.29e-01
),
cutip_gsf = cms.vdouble(
1.09e-02, 9.10e-03, 2.49e-02, 6.42e-03, 2.82e-02, 2.77e-02, 5.72e-03, 4.03e-02, -5.13e-03
),
cutip_gsfl = cms.vdouble(
2.57e-02, 9.56e-02, 1.19e-01, 7.51e-04, 8.18e-01, 6.35e-01, -2.05e-03, 5.44e+00, 2.61e-02
),
cutsee = cms.vdouble(
1.03e-02, 9.96e-03, 9.85e-03, 2.78e-02, 2.67e-02, 2.65e-02, 9.31e-03, 2.31e-02, -2.14e-02
),
cutseel = cms.vdouble(
1.44e-02, 1.22e-02, 1.59e-02, 4.02e-02, 4.05e-02, 4.44e-02, 1.53e-02, 4.34e-02, 5.64e-02
)
)

eidHyperTight3 = eidCutBasedExt.clone()
eidHyperTight3.electronIDType = 'classbased'
eidHyperTight3.electronQuality = 'hypertight3'
eidHyperTight3.electronVersion = 'V06'
eidHyperTight3.additionalCategories = True
eidHyperTight3.classbasedhypertight3EleIDCutsV06 = cms.PSet(
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdcotdist = cms.vdouble(
1.14e-03, 4.86e-06, 7.17e-05, 2.50e-05, 4.67e-04, 1.86e-05, 7.27e-06, 2.38e-04, 9.18e-06
),
cutdetain = cms.vdouble(
3.20e-03, 1.30e-03, 3.70e-03, 2.90e-03, 3.48e-03, 3.92e-03, 3.75e-03, 1.48e-02, -4.46e-04
),
cutdetainl = cms.vdouble(
1.83e-03, -3.15e-04, 2.25e-02, 1.35e-02, 3.38e-02, 1.85e-02, -8.90e-03, 1.96e-01, 5.06e-03
),
cutdphiin = cms.vdouble(
1.32e-02, 1.21e-02, 1.60e-02, 1.30e-02, 5.56e-03, 7.07e-03, 1.47e-02, 2.30e-02, 3.30e-01
),
cutdphiinl = cms.vdouble(
1.49e-02, 1.06e-01, 4.66e-02, 7.44e-02, 7.66e-02, 2.96e-02, 3.96e-02, 6.07e-01, 8.35e-02
),
cuteseedopcor = cms.vdouble(
1.07e+00, 1.01e+00, 1.12e+00, 1.14e+00, 9.56e-01, 1.21e+00, 1.05e+00, 1.34e+00, 6.11e-01
),
cutfmishits = cms.vdouble(
5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, -5.00e-01, -5.00e-01, 5.00e-01, -5.00e-01, -5.00e-01
),
cuthoe = cms.vdouble(
1.39e-02, 1.81e-02, 5.16e-03, 1.50e-02, 6.04e-03, 1.29e-02, 2.78e-02, 4.88e-02, -3.64e-02
),
cuthoel = cms.vdouble(
-4.11e-02, 5.73e-02, -4.16e-02, 7.09e-02, -1.35e-02, 6.30e-02, 1.24e-01, -1.51e-01, 3.29e-01
),
cutip_gsf = cms.vdouble(
8.77e-03, 7.26e-03, 1.74e-02, 6.24e-03, 2.82e-02, 2.77e-02, 5.72e-03, 4.03e-02, -5.13e-03
),
cutip_gsfl = cms.vdouble(
2.57e-02, 9.56e-02, 1.19e-01, 7.51e-04, 8.18e-01, 6.35e-01, -2.05e-03, 5.44e+00, 2.61e-02
),
cutsee = cms.vdouble(
1.03e-02, 9.66e-03, 9.79e-03, 2.68e-02, 2.67e-02, 2.65e-02, 9.29e-03, 2.31e-02, -2.14e-02
),
cutseel = cms.vdouble(
1.44e-02, 1.22e-02, 1.59e-02, 3.99e-02, 4.05e-02, 4.44e-02, 1.53e-02, 4.34e-02, 5.64e-02
)
)

eidHyperTight4 = eidCutBasedExt.clone()
eidHyperTight4.electronIDType = 'classbased'
eidHyperTight4.electronQuality = 'hypertight4'
eidHyperTight4.electronVersion = 'V06'
eidHyperTight4.additionalCategories = True
eidHyperTight4.classbasedhypertight4EleIDCutsV06 = cms.PSet(
cutiso_sum = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoet = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutiso_sumoetl = cms.vdouble(
99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999., 99999.
),
cutdcotdist = cms.vdouble(
8.87e-05, 4.86e-06, 5.58e-06, 2.50e-05, 4.67e-04, 1.86e-05, 5.65e-07, 2.38e-04, 9.18e-06
),
cutdetain = cms.vdouble(
1.66e-03, 1.30e-03, 9.47e-04, 2.90e-03, 3.48e-03, 3.92e-03, 2.49e-03, 1.48e-02, -4.46e-04
),
cutdetainl = cms.vdouble(
2.20e-04, -3.15e-04, 2.07e-02, 1.35e-02, 3.38e-02, 1.85e-02, -1.01e-02, 1.96e-01, 5.06e-03
),
cutdphiin = cms.vdouble(
5.02e-03, 1.21e-02, 6.15e-03, 1.30e-02, 5.56e-03, 7.07e-03, 4.37e-03, 2.30e-02, 3.30e-01
),
cutdphiinl = cms.vdouble(
5.13e-03, 1.06e-01, 3.85e-02, 7.44e-02, 7.66e-02, 2.96e-02, 3.02e-02, 6.07e-01, 8.35e-02
),
cuteseedopcor = cms.vdouble(
1.15e+00, 1.01e+00, 1.23e+00, 1.14e+00, 9.56e-01, 1.21e+00, 1.05e+00, 1.34e+00, 6.11e-01
),
cutfmishits = cms.vdouble(
5.00e-01, -5.00e-01, 5.00e-01, -5.00e-01, -5.00e-01, -5.00e-01, 5.00e-01, -5.00e-01, -5.00e-01
),
cuthoe = cms.vdouble(
1.18e-02, 1.81e-02, 5.16e-03, 1.50e-02, 6.04e-03, 1.29e-02, 2.78e-02, 4.88e-02, -3.64e-02
),
cuthoel = cms.vdouble(
-4.67e-02, 5.73e-02, -4.16e-02, 7.09e-02, -1.35e-02, 6.30e-02, 1.24e-01, -1.51e-01, 3.29e-01
),
cutip_gsf = cms.vdouble(
4.39e-03, 7.26e-03, 4.08e-03, 6.24e-03, 2.82e-02, 2.77e-02, 3.07e-03, 4.03e-02, -5.13e-03
),
cutip_gsfl = cms.vdouble(
2.41e-02, 9.56e-02, 1.16e-01, 7.51e-04, 8.18e-01, 6.35e-01, -2.05e-03, 5.44e+00, 2.61e-02
),
cutsee = cms.vdouble(
9.39e-03, 9.66e-03, 8.87e-03, 2.68e-02, 2.67e-02, 2.65e-02, 9.13e-03, 2.31e-02, -2.14e-02
),
cutseel = cms.vdouble(
1.41e-02, 1.22e-02, 1.59e-02, 3.99e-02, 4.05e-02, 4.44e-02, 1.53e-02, 4.34e-02, 5.64e-02
)
)

