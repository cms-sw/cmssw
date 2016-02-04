import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi import *

eidVeryLoose = eidCutBasedExt.clone()
eidVeryLoose.electronIDType = 'classbased'
eidVeryLoose.electronQuality = 'veryloose'
eidVeryLoose.electronVersion = 'V06'
eidVeryLoose.additionalCategories = True
eidVeryLoose.classbasedverylooseEleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(
3.89e-02, 3.90e-02, 3.96e-02, 3.92e-02, 3.95e-02, 3.97e-02, 3.92e-02, 3.95e-02, 2.98e-02
),
cutdetain = cms.vdouble(
1.33e-02, 6.97e-03, 2.43e-02, 2.47e-02, 5.50e-02, 2.20e-02, 4.31e-02, 3.84e-02, 3.13e-02
),
cutdetainl = cms.vdouble(
1.28e-02, 5.93e-03, 2.64e-02, 2.72e-02, 6.72e-02, 2.06e-02, 1.92e-02, 1.97e-01, 2.91e-02
),
cutdphiin = cms.vdouble(
9.71e-02, 2.70e-01, 3.59e-01, 8.36e-02, 4.42e-01, 3.34e-01, 3.63e-01, 4.04e-01, 9.20e-01
),
cutdphiinl = cms.vdouble(
7.93e-02, 2.66e-01, 3.60e-01, 9.12e-02, 4.42e-01, 3.33e-01, 3.39e-01, 6.61e-01, 2.92e-01
),
cuteseedopcor = cms.vdouble(
6.35e-01, 3.27e-01, 4.00e-01, 7.31e-01, 3.50e-01, 4.54e-01, 1.27e-01, 2.91e-01, 6.28e-02
),
cutfmishits = cms.vdouble(
4.50e+00, 1.50e+00, 1.50e+00, 4.50e+00, 2.50e+00, 1.50e+00, 3.50e+00, 4.50e+00, 3.50e+00
),
cuthoe = cms.vdouble(
2.30e-01, 1.16e-01, 1.48e-01, 3.66e-01, 1.01e-01, 1.46e-01, 4.29e-01, 4.42e-01, 4.00e-01
),
cuthoel = cms.vdouble(
2.44e-01, 1.17e-01, 1.48e-01, 3.60e-01, 7.69e-02, 1.46e-01, 3.26e-01, 3.83e-01, 3.93e-01
),
cutip_gsf = cms.vdouble(
8.48e-02, 1.05e-01, 1.78e-01, 8.78e-02, 7.13e-01, 4.77e-01, 4.30e-01, 5.69e+00, 4.76e-01
),
cutip_gsfl = cms.vdouble(
8.70e-02, 1.09e-01, 1.79e-01, 7.55e-02, 7.14e-01, 5.24e-01, 9.01e-01, 1.84e+00, 3.01e-01
),
cutiso_sum = cms.vdouble(
2.56e+01, 1.70e+01, 1.76e+01, 1.86e+01, 8.79e+00, 1.25e+01, 2.14e+01, 2.34e+01, 3.23e+00
),
cutiso_sumoet = cms.vdouble(
5.38e+01, 1.07e+01, 1.03e+01, 4.02e+01, 5.81e+00, 8.01e+00, 9.27e+00, 1.15e+01, 8.86e+02
),
cutiso_sumoetl = cms.vdouble(
1.76e+01, 1.10e+01, 1.14e+01, 1.37e+01, 6.28e+00, 8.27e+00, 1.59e+01, 1.58e+01, 8.08e+00
),
cutsee = cms.vdouble(
1.57e-02, 1.20e-02, 1.84e-02, 3.98e-02, 3.24e-02, 3.81e-02, 1.25e-02, 6.42e-02, 6.69e-02
),
cutseel = cms.vdouble(
1.77e-02, 1.23e-02, 1.92e-02, 4.73e-02, 3.54e-02, 4.87e-02, 1.59e-02, 6.17e-02, 1.19e-01
)
)

eidLoose = eidCutBasedExt.clone()
eidLoose.electronIDType = 'classbased'
eidLoose.electronQuality = 'loose'
eidLoose.electronVersion = 'V06'
eidLoose.additionalCategories = True
eidLoose.classbasedlooseEleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(
3.87e-02, 3.50e-02, 3.18e-02, 3.92e-02, 3.94e-02, 3.97e-02, 3.10e-02, 3.95e-02, 1.10e-02
),
cutdetain = cms.vdouble(
1.33e-02, 5.28e-03, 1.44e-02, 2.19e-02, 1.25e-02, 1.37e-02, 2.18e-02, 3.84e-02, 2.75e-02
),
cutdetainl = cms.vdouble(
1.26e-02, 4.88e-03, 1.68e-02, 2.67e-02, 1.21e-02, 1.31e-02, 1.92e-02, 1.97e-01, 2.84e-02
),
cutdphiin = cms.vdouble(
9.36e-02, 2.46e-01, 3.25e-01, 8.18e-02, 3.22e-01, 2.83e-01, 3.54e-01, 4.04e-01, 6.80e-01
),
cutdphiinl = cms.vdouble(
7.93e-02, 2.44e-01, 3.11e-01, 9.12e-02, 3.04e-01, 2.82e-01, 3.39e-01, 6.61e-01, 2.91e-01
),
cuteseedopcor = cms.vdouble(
6.37e-01, 8.79e-01, 4.02e-01, 7.45e-01, 3.67e-01, 4.88e-01, 1.27e-01, 7.19e-01, 6.28e-02
),
cutfmishits = cms.vdouble(
4.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 3.50e+00, 3.50e+00, 3.50e+00
),
cuthoe = cms.vdouble(
1.96e-01, 7.92e-02, 1.48e-01, 3.66e-01, 6.88e-02, 1.45e-01, 4.29e-01, 4.42e-01, 4.00e-01
),
cuthoel = cms.vdouble(
2.26e-01, 7.95e-02, 1.48e-01, 3.60e-01, 6.23e-02, 1.46e-01, 3.26e-01, 3.83e-01, 3.92e-01
),
cutip_gsf = cms.vdouble(
8.48e-02, 9.95e-02, 1.75e-01, 6.97e-02, 5.65e-01, 4.77e-01, 4.30e-01, 3.32e+00, 1.61e-01
),
cutip_gsfl = cms.vdouble(
7.58e-02, 9.81e-02, 1.76e-01, 6.66e-02, 5.65e-01, 5.16e-01, 9.01e-01, 1.12e+00, 8.42e-02
),
cutiso_sum = cms.vdouble(
2.02e+01, 1.31e+01, 1.56e+01, 1.61e+01, 8.61e+00, 1.10e+01, 1.31e+01, 1.63e+01, 2.37e+00
),
cutiso_sumoet = cms.vdouble(
1.49e+01, 8.33e+00, 7.64e+00, 1.22e+01, 4.49e+00, 5.59e+00, 7.44e+00, 7.31e+00, 2.74e+01
),
cutiso_sumoetl = cms.vdouble(
1.21e+01, 9.18e+00, 8.66e+00, 9.43e+00, 4.34e+00, 5.73e+00, 1.08e+01, 1.05e+01, 8.08e+00
),
cutsee = cms.vdouble(
1.57e-02, 1.12e-02, 1.40e-02, 3.95e-02, 3.10e-02, 3.37e-02, 1.11e-02, 6.13e-02, 6.69e-02
),
cutseel = cms.vdouble(
1.77e-02, 1.15e-02, 1.50e-02, 4.55e-02, 3.24e-02, 4.46e-02, 1.22e-02, 6.17e-02, 1.19e-01
)
)

eidMedium = eidCutBasedExt.clone()
eidMedium.electronIDType = 'classbased'
eidMedium.electronQuality = 'medium'
eidMedium.electronVersion = 'V06'
eidMedium.additionalCategories = True
eidMedium.classbasedmediumEleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(
3.32e-02, 2.92e-02, 2.49e-02, 3.92e-02, 3.41e-02, 3.96e-02, 2.91e-02, 3.95e-02, 7.71e-03
),
cutdetain = cms.vdouble(
1.33e-02, 4.48e-03, 9.22e-03, 1.54e-02, 7.26e-03, 1.24e-02, 1.29e-02, 3.84e-02, 1.88e-02
),
cutdetainl = cms.vdouble(
1.21e-02, 4.22e-03, 9.18e-03, 1.61e-02, 6.45e-03, 1.16e-02, 1.23e-02, 6.20e-02, 2.43e-02
),
cutdphiin = cms.vdouble(
7.09e-02, 2.43e-01, 2.96e-01, 7.98e-02, 2.35e-01, 2.76e-01, 3.42e-01, 4.04e-01, 2.99e-01
),
cutdphiinl = cms.vdouble(
7.42e-02, 2.43e-01, 2.97e-01, 9.12e-02, 2.26e-01, 2.76e-01, 3.34e-01, 5.58e-01, 2.91e-01
),
cuteseedopcor = cms.vdouble(
6.42e-01, 9.44e-01, 4.53e-01, 7.62e-01, 3.67e-01, 5.57e-01, 1.98e-01, 9.15e-01, 6.28e-02
),
cutfmishits = cms.vdouble(
4.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 5.00e-01, 1.50e+00, 5.00e-01, 5.00e-01
),
cuthoe = cms.vdouble(
1.96e-01, 6.30e-02, 1.48e-01, 3.66e-01, 5.66e-02, 1.45e-01, 4.29e-01, 4.28e-01, 3.99e-01
),
cuthoel = cms.vdouble(
2.19e-01, 6.19e-02, 1.47e-01, 3.58e-01, 4.61e-02, 1.46e-01, 3.26e-01, 3.81e-01, 3.89e-01
),
cutip_gsf = cms.vdouble(
2.45e-02, 9.74e-02, 1.48e-01, 5.49e-02, 5.65e-01, 3.33e-01, 2.04e-01, 5.41e-01, 1.21e-01
),
cutip_gsfl = cms.vdouble(
1.92e-02, 9.81e-02, 1.33e-01, 4.34e-02, 5.65e-01, 3.24e-01, 2.33e-01, 4.30e-01, 6.44e-02
),
cutiso_sum = cms.vdouble(
1.44e+01, 1.12e+01, 1.09e+01, 1.08e+01, 6.35e+00, 9.78e+00, 1.30e+01, 1.62e+01, 1.96e+00
),
cutiso_sumoet = cms.vdouble(
1.01e+01, 6.41e+00, 6.00e+00, 8.14e+00, 3.90e+00, 4.76e+00, 6.86e+00, 6.48e+00, 1.74e+01
),
cutiso_sumoetl = cms.vdouble(
9.44e+00, 7.67e+00, 7.15e+00, 7.34e+00, 3.35e+00, 4.70e+00, 8.32e+00, 7.55e+00, 6.25e+00
),
cutsee = cms.vdouble(
1.30e-02, 1.09e-02, 1.18e-02, 3.94e-02, 3.04e-02, 3.28e-02, 1.00e-02, 3.73e-02, 6.69e-02
),
cutseel = cms.vdouble(
1.42e-02, 1.11e-02, 1.29e-02, 4.32e-02, 2.96e-02, 3.82e-02, 1.01e-02, 4.45e-02, 1.19e-01
)
)

eidTight = eidCutBasedExt.clone()
eidTight.electronIDType = 'classbased'
eidTight.electronQuality = 'tight'
eidTight.electronVersion = 'V06'
eidTight.additionalCategories = True
eidTight.classbasedtightEleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(
2.68e-02, 2.36e-02, 2.21e-02, 3.72e-02, 3.17e-02, 3.61e-02, 2.55e-02, 3.75e-02, 2.16e-04
),
cutdetain = cms.vdouble(
8.92e-03, 3.96e-03, 8.50e-03, 1.34e-02, 6.27e-03, 1.05e-02, 1.12e-02, 3.09e-02, 1.88e-02
),
cutdetainl = cms.vdouble(
9.23e-03, 3.77e-03, 8.70e-03, 1.39e-02, 5.60e-03, 9.40e-03, 1.07e-02, 6.20e-02, 4.10e-03
),
cutdphiin = cms.vdouble(
6.37e-02, 1.53e-01, 2.90e-01, 7.69e-02, 1.81e-01, 2.34e-01, 3.42e-01, 3.93e-01, 2.84e-01
),
cutdphiinl = cms.vdouble(
6.92e-02, 2.33e-01, 2.96e-01, 8.65e-02, 1.85e-01, 2.76e-01, 3.34e-01, 3.53e-01, 2.90e-01
),
cuteseedopcor = cms.vdouble(
6.52e-01, 9.69e-01, 9.12e-01, 7.79e-01, 3.67e-01, 6.99e-01, 3.28e-01, 9.67e-01, 5.89e-01
),
cutfmishits = cms.vdouble(
4.50e+00, 1.50e+00, 5.00e-01, 1.50e+00, 1.50e+00, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01
),
cuthoe = cms.vdouble(
1.74e-01, 4.88e-02, 1.46e-01, 3.64e-01, 4.93e-02, 1.45e-01, 4.29e-01, 4.20e-01, 3.99e-01
),
cuthoel = cms.vdouble(
2.19e-01, 5.25e-02, 1.47e-01, 3.57e-01, 4.25e-02, 1.45e-01, 3.26e-01, 3.80e-01, 1.32e-01
),
cutip_gsf = cms.vdouble(
1.58e-02, 8.25e-02, 1.15e-01, 4.05e-02, 5.40e-01, 1.51e-01, 7.74e-02, 4.17e-01, 7.80e-02
),
cutip_gsfl = cms.vdouble(
1.27e-02, 6.26e-02, 9.68e-02, 3.02e-02, 5.65e-01, 1.46e-01, 7.90e-02, 4.10e-01, 4.79e-02
),
cutiso_sum = cms.vdouble(
1.23e+01, 9.77e+00, 1.01e+01, 9.77e+00, 6.13e+00, 7.55e+00, 1.30e+01, 1.62e+01, 1.78e+00
),
cutiso_sumoet = cms.vdouble(
7.75e+00, 5.45e+00, 5.67e+00, 5.97e+00, 3.17e+00, 3.86e+00, 6.06e+00, 5.31e+00, 1.05e+01
),
cutiso_sumoetl = cms.vdouble(
7.56e+00, 5.08e+00, 5.77e+00, 5.74e+00, 2.37e+00, 3.32e+00, 4.97e+00, 5.46e+00, 3.82e+00
),
cutsee = cms.vdouble(
1.16e-02, 1.07e-02, 1.08e-02, 3.49e-02, 2.89e-02, 3.08e-02, 9.87e-03, 3.37e-02, 4.40e-02
),
cutseel = cms.vdouble(
1.27e-02, 1.08e-02, 1.13e-02, 4.19e-02, 2.81e-02, 3.02e-02, 9.76e-03, 4.28e-02, 2.98e-02
)
)

eidSuperTight = eidCutBasedExt.clone()
eidSuperTight.electronIDType = 'classbased'
eidSuperTight.electronQuality = 'supertight'
eidSuperTight.electronVersion = 'V06'
eidSuperTight.additionalCategories = True
eidSuperTight.classbasedsupertightEleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(
2.11e-02, 1.86e-02, 1.55e-02, 3.40e-02, 2.85e-02, 3.32e-02, 1.64e-02, 3.75e-02, 1.30e-04
),
cutdetain = cms.vdouble(
7.84e-03, 3.67e-03, 7.00e-03, 1.28e-02, 5.65e-03, 9.53e-03, 1.08e-02, 2.97e-02, 7.24e-03
),
cutdetainl = cms.vdouble(
7.61e-03, 3.28e-03, 6.57e-03, 1.03e-02, 5.05e-03, 8.55e-03, 1.07e-02, 2.94e-02, 4.10e-03
),
cutdphiin = cms.vdouble(
4.83e-02, 7.39e-02, 2.38e-01, 5.74e-02, 1.29e-01, 2.13e-01, 3.31e-01, 3.93e-01, 2.84e-01
),
cutdphiinl = cms.vdouble(
5.79e-02, 7.21e-02, 2.18e-01, 7.70e-02, 1.41e-01, 2.11e-01, 2.43e-01, 3.53e-01, 2.89e-01
),
cuteseedopcor = cms.vdouble(
7.32e-01, 9.77e-01, 9.83e-01, 8.55e-01, 4.31e-01, 7.35e-01, 4.18e-01, 9.99e-01, 5.89e-01
),
cutfmishits = cms.vdouble(
3.50e+00, 1.50e+00, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01
),
cuthoe = cms.vdouble(
9.19e-02, 4.11e-02, 1.42e-01, 3.35e-01, 3.82e-02, 1.41e-01, 4.29e-01, 4.01e-01, 3.99e-01
),
cuthoel = cms.vdouble(
7.51e-02, 3.81e-02, 1.41e-01, 3.32e-01, 3.10e-02, 1.43e-01, 2.35e-01, 3.80e-01, 1.32e-01
),
cutip_gsf = cms.vdouble(
1.42e-02, 2.66e-02, 1.06e-01, 3.38e-02, 3.23e-01, 1.07e-01, 7.74e-02, 2.32e-01, 7.80e-02
),
cutip_gsfl = cms.vdouble(
1.15e-02, 2.72e-02, 8.41e-02, 2.49e-02, 4.17e-01, 1.02e-01, 7.90e-02, 1.69e-01, 4.79e-02
),
cutiso_sum = cms.vdouble(
8.95e+00, 8.18e+00, 8.75e+00, 7.47e+00, 5.43e+00, 5.87e+00, 8.16e+00, 1.02e+01, 1.78e+00
),
cutiso_sumoet = cms.vdouble(
6.45e+00, 5.14e+00, 4.99e+00, 5.21e+00, 2.65e+00, 3.12e+00, 4.52e+00, 4.72e+00, 3.68e+00
),
cutiso_sumoetl = cms.vdouble(
6.02e+00, 3.96e+00, 4.23e+00, 4.73e+00, 1.99e+00, 2.64e+00, 3.72e+00, 3.81e+00, 1.44e+00
),
cutsee = cms.vdouble(
1.09e-02, 1.05e-02, 1.05e-02, 3.24e-02, 2.81e-02, 2.95e-02, 9.77e-03, 2.75e-02, 2.95e-02
),
cutseel = cms.vdouble(
1.12e-02, 1.05e-02, 1.07e-02, 3.51e-02, 2.75e-02, 2.87e-02, 9.59e-03, 2.67e-02, 2.98e-02
)
)

eidHyperTight1 = eidCutBasedExt.clone()
eidHyperTight1.electronIDType = 'classbased'
eidHyperTight1.electronQuality = 'hypertight1'
eidHyperTight1.electronVersion = 'V06'
eidHyperTight1.additionalCategories = True
eidHyperTight1.classbasedhypertight1EleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(
1.48e-02, 1.50e-02, 8.25e-03, 3.16e-02, 2.85e-02, 3.15e-02, 6.62e-03, 3.48e-02, 3.63e-06
),
cutdetain = cms.vdouble(
6.51e-03, 3.51e-03, 5.53e-03, 9.16e-03, 5.30e-03, 8.28e-03, 1.08e-02, 2.97e-02, 7.24e-03
),
cutdetainl = cms.vdouble(
6.05e-03, 3.23e-03, 4.93e-03, 8.01e-03, 4.93e-03, 7.91e-03, 1.03e-02, 2.94e-02, 4.10e-03
),
cutdphiin = cms.vdouble(
4.83e-02, 4.91e-02, 2.30e-01, 3.48e-02, 7.44e-02, 2.04e-01, 9.95e-02, 3.93e-01, 2.84e-01
),
cutdphiinl = cms.vdouble(
4.74e-02, 4.51e-02, 2.18e-01, 2.99e-02, 7.37e-02, 2.11e-01, 9.99e-02, 3.53e-01, 2.89e-01
),
cuteseedopcor = cms.vdouble(
7.72e-01, 9.90e-01, 1.01e+00, 8.55e-01, 9.11e-01, 7.72e-01, 9.17e-01, 1.06e+00, 7.63e-01
),
cutfmishits = cms.vdouble(
3.50e+00, 1.50e+00, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01
),
cuthoe = cms.vdouble(
6.17e-02, 3.70e-02, 1.41e-01, 2.91e-01, 3.82e-02, 1.34e-01, 4.19e-01, 3.87e-01, 3.93e-01
),
cuthoel = cms.vdouble(
4.43e-02, 3.57e-02, 1.41e-01, 2.81e-01, 3.07e-02, 1.28e-01, 2.27e-01, 3.80e-01, 1.32e-01
),
cutip_gsf = cms.vdouble(
1.21e-02, 1.76e-02, 6.01e-02, 2.96e-02, 1.74e-01, 9.70e-02, 7.74e-02, 1.33e-01, 7.80e-02
),
cutip_gsfl = cms.vdouble(
1.01e-02, 1.56e-02, 6.87e-02, 2.13e-02, 1.25e-01, 8.16e-02, 7.90e-02, 1.30e-01, 4.79e-02
),
cutiso_sum = cms.vdouble(
7.92e+00, 6.85e+00, 7.87e+00, 6.77e+00, 4.47e+00, 5.28e+00, 6.57e+00, 1.02e+01, 1.78e+00
),
cutiso_sumoet = cms.vdouble(
5.20e+00, 3.93e+00, 3.88e+00, 4.10e+00, 2.40e+00, 2.43e+00, 3.49e+00, 3.94e+00, 3.01e+00
),
cutiso_sumoetl = cms.vdouble(
4.18e+00, 3.12e+00, 3.44e+00, 3.25e+00, 1.77e+00, 2.06e+00, 2.83e+00, 3.12e+00, 1.43e+00
),
cutsee = cms.vdouble(
1.05e-02, 1.04e-02, 1.01e-02, 3.24e-02, 2.80e-02, 2.85e-02, 9.67e-03, 2.61e-02, 2.95e-02
),
cutseel = cms.vdouble(
1.04e-02, 1.03e-02, 1.01e-02, 3.04e-02, 2.74e-02, 2.78e-02, 9.58e-03, 2.54e-02, 2.83e-02
)
)

eidHyperTight2 = eidCutBasedExt.clone()
eidHyperTight2.electronIDType = 'classbased'
eidHyperTight2.electronQuality = 'hypertight2'
eidHyperTight2.electronVersion = 'V06'
eidHyperTight2.additionalCategories = True
eidHyperTight2.classbasedhypertight2EleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(
1.15e-02, 1.07e-02, 4.01e-03, 2.97e-02, 2.85e-02, 3.10e-02, 9.34e-04, 3.40e-02, 2.82e-07
),
cutdetain = cms.vdouble(
5.29e-03, 2.56e-03, 4.89e-03, 7.89e-03, 5.30e-03, 7.37e-03, 8.91e-03, 9.36e-03, 5.94e-03
),
cutdetainl = cms.vdouble(
4.48e-03, 2.59e-03, 4.42e-03, 6.54e-03, 4.93e-03, 6.98e-03, 8.49e-03, 9.06e-03, -4.81e-03
),
cutdphiin = cms.vdouble(
2.41e-02, 3.83e-02, 1.48e-01, 2.91e-02, 3.15e-02, 1.57e-01, 8.90e-02, 1.02e-01, 2.81e-01
),
cutdphiinl = cms.vdouble(
2.13e-02, 3.79e-02, 1.25e-01, 2.24e-02, 3.69e-02, 1.64e-01, 9.99e-02, 9.23e-02, 2.37e-01
),
cuteseedopcor = cms.vdouble(
1.03e+00, 9.95e-01, 1.03e+00, 1.01e+00, 9.46e-01, 9.03e-01, 9.97e-01, 1.14e+00, 8.00e-01
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, -5.00e-01
),
cuthoe = cms.vdouble(
4.94e-02, 3.45e-02, 1.40e-01, 2.02e-01, 3.82e-02, 1.19e-01, 1.23e-01, 3.82e-01, 2.50e-01
),
cuthoel = cms.vdouble(
4.04e-02, 3.42e-02, 1.31e-01, 1.85e-01, 3.01e-02, 1.27e-01, 2.27e-01, 3.80e-01, 1.32e-01
),
cutip_gsf = cms.vdouble(
1.14e-02, 1.38e-02, 5.29e-02, 1.87e-02, 1.31e-01, 8.63e-02, 7.74e-02, 1.04e-01, 2.42e-02
),
cutip_gsfl = cms.vdouble(
9.83e-03, 1.35e-02, 4.27e-02, 1.72e-02, 1.25e-01, 7.92e-02, 7.90e-02, 1.30e-01, 3.40e-02
),
cutiso_sum = cms.vdouble(
6.40e+00, 5.77e+00, 6.54e+00, 5.22e+00, 3.86e+00, 4.63e+00, 6.31e+00, 1.02e+01, 1.78e+00
),
cutiso_sumoet = cms.vdouble(
4.03e+00, 3.03e+00, 3.24e+00, 3.13e+00, 2.05e+00, 2.01e+00, 2.99e+00, 3.44e+00, 2.76e+00
),
cutiso_sumoetl = cms.vdouble(
3.08e+00, 2.31e+00, 2.84e+00, 2.53e+00, 1.65e+00, 1.72e+00, 2.34e+00, 3.11e+00, 1.35e+00
),
cutsee = cms.vdouble(
1.03e-02, 1.03e-02, 9.88e-03, 3.03e-02, 2.79e-02, 2.79e-02, 9.67e-03, 2.52e-02, 2.58e-02
),
cutseel = cms.vdouble(
1.02e-02, 1.02e-02, 9.80e-03, 2.90e-02, 2.74e-02, 2.75e-02, 9.58e-03, 2.49e-02, 2.50e-02
)
)

eidHyperTight3 = eidCutBasedExt.clone()
eidHyperTight3.electronIDType = 'classbased'
eidHyperTight3.electronQuality = 'hypertight3'
eidHyperTight3.electronVersion = 'V06'
eidHyperTight3.additionalCategories = True
eidHyperTight3.classbasedhypertight3EleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(
9.63e-03, 5.11e-03, 1.95e-04, 2.97e-02, 2.85e-02, 2.18e-02, 2.61e-05, 2.57e-02, 2.82e-07
),
cutdetain = cms.vdouble(
4.86e-03, 2.29e-03, 4.40e-03, 7.79e-03, 4.07e-03, 6.33e-03, 7.70e-03, 7.93e-03, 5.94e-03
),
cutdetainl = cms.vdouble(
4.48e-03, 2.30e-03, 4.14e-03, 6.04e-03, 3.87e-03, 6.09e-03, 7.97e-03, 8.04e-03, -4.81e-03
),
cutdphiin = cms.vdouble(
2.41e-02, 2.88e-02, 7.39e-02, 2.91e-02, 1.91e-02, 1.14e-01, 3.61e-02, 8.92e-02, 2.81e-01
),
cutdphiinl = cms.vdouble(
1.95e-02, 3.42e-02, 8.06e-02, 2.22e-02, 2.26e-02, 9.73e-02, 4.51e-02, 9.23e-02, 2.37e-01
),
cuteseedopcor = cms.vdouble(
1.07e+00, 1.01e+00, 1.08e+00, 1.01e+00, 9.69e-01, 9.10e-01, 1.04e+00, 1.20e+00, 8.00e-01
),
cutfmishits = cms.vdouble(
5.00e-01, 1.50e+00, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, -5.00e-01
),
cuthoe = cms.vdouble(
3.52e-02, 3.45e-02, 1.33e-01, 1.88e-01, 2.72e-02, 1.19e-01, 9.28e-02, 2.46e-01, 2.50e-01
),
cuthoel = cms.vdouble(
4.04e-02, 3.40e-02, 1.31e-01, 1.84e-01, 2.64e-02, 1.18e-01, 9.76e-02, 2.53e-01, 1.32e-01
),
cutip_gsf = cms.vdouble(
1.14e-02, 1.26e-02, 3.79e-02, 1.68e-02, 1.21e-01, 5.29e-02, 7.74e-02, 3.35e-02, 2.42e-02
),
cutip_gsfl = cms.vdouble(
9.83e-03, 1.18e-02, 3.59e-02, 1.56e-02, 1.20e-01, 5.36e-02, 7.90e-02, 2.88e-02, 3.40e-02
),
cutiso_sum = cms.vdouble(
5.40e+00, 5.41e+00, 5.88e+00, 4.32e+00, 3.86e+00, 4.33e+00, 5.87e+00, 9.05e+00, 1.78e+00
),
cutiso_sumoet = cms.vdouble(
3.03e+00, 2.50e+00, 2.58e+00, 2.44e+00, 1.91e+00, 1.76e+00, 2.92e+00, 3.13e+00, 2.76e+00
),
cutiso_sumoetl = cms.vdouble(
2.36e+00, 2.02e+00, 2.29e+00, 1.89e+00, 1.65e+00, 1.69e+00, 2.03e+00, 2.79e+00, 1.35e+00
),
cutsee = cms.vdouble(
1.03e-02, 1.01e-02, 9.84e-03, 2.89e-02, 2.74e-02, 2.73e-02, 9.47e-03, 2.44e-02, 2.58e-02
),
cutseel = cms.vdouble(
1.02e-02, 1.00e-02, 9.73e-03, 2.79e-02, 2.73e-02, 2.69e-02, 9.40e-03, 2.46e-02, 2.50e-02
)
)

eidHyperTight4 = eidCutBasedExt.clone()
eidHyperTight4.electronIDType = 'classbased'
eidHyperTight4.electronQuality = 'hypertight4'
eidHyperTight4.electronVersion = 'V06'
eidHyperTight4.additionalCategories = True
eidHyperTight4.classbasedhypertight4EleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(
2.70e-04, 1.43e-04, 1.95e-04, 2.64e-03, 2.82e-02, 1.64e-02, 2.61e-05, 2.57e-02, 2.82e-07
),
cutdetain = cms.vdouble(
2.44e-03, 1.67e-03, 2.26e-03, 3.43e-03, 3.51e-03, 3.52e-03, 2.98e-03, 4.79e-03, 5.94e-03
),
cutdetainl = cms.vdouble(
2.34e-03, 1.29e-03, 2.30e-03, 3.30e-03, 3.61e-03, 3.84e-03, 2.53e-03, 3.66e-03, -4.81e-03
),
cutdphiin = cms.vdouble(
8.44e-03, 5.21e-03, 2.18e-02, 1.39e-02, 7.82e-03, 1.52e-02, 2.59e-02, 3.87e-02, 2.81e-01
),
cutdphiinl = cms.vdouble(
5.77e-03, 3.20e-03, 2.85e-02, 2.22e-02, 7.00e-03, 1.84e-02, 2.91e-02, 4.40e-02, 2.37e-01
),
cuteseedopcor = cms.vdouble(
1.15e+00, 1.01e+00, 1.21e+00, 1.07e+00, 9.69e-01, 9.10e-01, 1.08e+00, 1.36e+00, 8.00e-01
),
cutfmishits = cms.vdouble(
5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, -5.00e-01
),
cuthoe = cms.vdouble(
2.39e-02, 2.68e-02, 2.12e-02, 1.03e-01, 9.92e-03, 7.07e-02, 7.12e-02, 1.48e-01, 2.50e-01
),
cuthoel = cms.vdouble(
2.87e-02, 1.94e-02, 2.16e-02, 5.68e-02, 1.35e-02, 4.04e-02, 7.98e-02, 1.50e-01, 1.32e-01
),
cutip_gsf = cms.vdouble(
7.61e-03, 5.22e-03, 3.79e-02, 1.02e-02, 4.62e-02, 1.82e-02, 7.74e-02, 3.35e-02, 2.42e-02
),
cutip_gsfl = cms.vdouble(
7.81e-03, 4.25e-03, 3.08e-02, 1.04e-02, 2.35e-02, 2.45e-02, 7.90e-02, 2.88e-02, 3.40e-02
),
cutiso_sum = cms.vdouble(
5.40e+00, 5.41e+00, 5.88e+00, 4.32e+00, 3.86e+00, 4.33e+00, 5.86e+00, 9.05e+00, 1.78e+00
),
cutiso_sumoet = cms.vdouble(
2.53e+00, 2.10e+00, 1.87e+00, 1.84e+00, 1.79e+00, 1.61e+00, 2.53e+00, 1.98e+00, 2.76e+00
),
cutiso_sumoetl = cms.vdouble(
2.28e+00, 2.02e+00, 2.04e+00, 1.69e+00, 1.65e+00, 1.61e+00, 2.03e+00, 1.82e+00, 1.35e+00
),
cutsee = cms.vdouble(
9.99e-03, 9.61e-03, 9.65e-03, 2.75e-02, 2.61e-02, 2.64e-02, 9.18e-03, 2.44e-02, 2.58e-02
),
cutseel = cms.vdouble(
9.66e-03, 9.69e-03, 9.58e-03, 2.73e-02, 2.66e-02, 2.66e-02, 8.64e-03, 2.46e-02, 2.50e-02
)
)

