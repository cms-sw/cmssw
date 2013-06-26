import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi import *

eidVeryLooseMC = eidCutBasedExt.clone()
eidVeryLooseMC.electronIDType = 'classbased'
eidVeryLooseMC.electronQuality = 'veryloose'
eidVeryLooseMC.electronVersion = 'V06'
eidVeryLooseMC.additionalCategories = True
eidVeryLooseMC.classbasedverylooseEleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutdetain = cms.vdouble(
1.37e-02, 9.33e-03, 2.57e-02, 2.92e-02, 5.14e-02, 2.89e-02, 4.00e-02, 3.08e-02, 3.20e-02
),
cutdetainl = cms.vdouble(
1.29e-02, 7.58e-03, 2.57e-02, 2.45e-02, 8.16e-02, 2.55e-02, 1.89e-02, 1.40e-01, 2.77e-02
),
cutdphiin = cms.vdouble(
8.97e-02, 2.86e-01, 3.62e-01, 1.16e-01, 4.52e-01, 3.45e-01, 3.62e-01, 4.04e-01, 6.83e-01
),
cutdphiinl = cms.vdouble(
7.73e-02, 2.83e-01, 3.63e-01, 9.57e-02, 4.58e-01, 3.46e-01, 3.33e-01, 6.47e-01, 2.92e-01
),
cuteseedopcor = cms.vdouble(
6.17e-01, 2.93e-01, 3.97e-01, 7.06e-01, 3.62e-01, 4.54e-01, 1.50e-01, 3.11e-01, 6.93e-02
),
cutfmishits = cms.vdouble(
4.50e+00, 1.50e+00, 1.50e+00, 6.50e+00, 2.50e+00, 3.50e+00, 4.50e+00, 4.50e+00, 4.50e+00
),
cuthoe = cms.vdouble(
2.50e-01, 1.55e-01, 1.47e-01, 3.71e-01, 1.10e-01, 1.48e-01, 5.20e-01, 4.52e-01, 4.04e-01
),
cuthoel = cms.vdouble(
2.78e-01, 1.26e-01, 1.47e-01, 3.75e-01, 6.77e-02, 1.45e-01, 3.67e-01, 3.83e-01, 3.92e-01
),
cutip_gsf = cms.vdouble(
5.51e-02, 8.58e-02, 1.43e-01, 8.74e-02, 6.02e-01, 6.04e-01, 1.42e-01, 1.15e+00, 2.36e-01
),
cutip_gsfl = cms.vdouble(
4.23e-02, 8.34e-02, 1.42e-01, 6.71e-02, 8.77e-01, 6.03e-01, 1.09e-01, 7.75e-01, 8.61e-02
),
cutiso_sum = cms.vdouble(
5.08e+01, 2.00e+01, 2.31e+01, 3.08e+01, 9.60e+00, 1.76e+01, 2.56e+01, 2.04e+01, 3.65e+00
),
cutiso_sumoet = cms.vdouble(
1.66e+02, 1.62e+01, 1.79e+01, 5.25e+01, 8.95e+00, 1.40e+01, 1.85e+01, 1.75e+01, 2.86e+01
),
cutiso_sumoetl = cms.vdouble(
3.86e+01, 1.09e+01, 1.29e+01, 1.09e+01, 6.49e+00, 8.72e+00, 1.33e+01, 1.34e+01, 7.59e+00
),
cutsee = cms.vdouble(
1.76e-02, 1.36e-02, 1.92e-02, 5.43e-02, 3.74e-02, 5.44e-02, 1.82e-02, 6.78e-02, 1.33e-01
),
cutseel = cms.vdouble(
1.76e-02, 1.22e-02, 1.92e-02, 5.23e-02, 3.47e-02, 4.90e-02, 2.19e-02, 6.12e-02, 5.51e-02
)
)

eidLooseMC = eidCutBasedExt.clone()
eidLooseMC.electronIDType = 'classbased'
eidLooseMC.electronQuality = 'loose'
eidLooseMC.electronVersion = 'V06'
eidLooseMC.additionalCategories = True
eidLooseMC.classbasedlooseEleIDCutsV06 = cms.PSet(
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

eidMediumMC = eidCutBasedExt.clone()
eidMediumMC.electronIDType = 'classbased'
eidMediumMC.electronQuality = 'medium'
eidMediumMC.electronVersion = 'V06'
eidMediumMC.additionalCategories = True
eidMediumMC.classbasedmediumEleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutdetain = cms.vdouble(
1.37e-02, 5.16e-03, 1.27e-02, 1.84e-02, 8.40e-03, 1.29e-02, 2.52e-02, 2.85e-02, 1.63e-02
),
cutdetainl = cms.vdouble(
1.19e-02, 4.29e-03, 1.26e-02, 1.90e-02, 6.39e-03, 1.24e-02, 1.71e-02, 9.17e-02, 1.49e-02
),
cutdphiin = cms.vdouble(
8.97e-02, 2.50e-01, 3.03e-01, 1.08e-01, 2.67e-01, 2.77e-01, 3.41e-01, 3.08e-01, 3.28e-01
),
cutdphiinl = cms.vdouble(
7.08e-02, 2.47e-01, 3.37e-01, 9.56e-02, 2.60e-01, 2.75e-01, 3.33e-01, 3.03e-01, 2.58e-01
),
cuteseedopcor = cms.vdouble(
6.37e-01, 9.05e-01, 4.79e-01, 7.35e-01, 7.30e-01, 5.48e-01, 2.14e-01, 8.73e-01, 3.73e-01
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 1.50e+00, 2.50e+00, 2.50e+00, 1.50e+00, 1.50e+00, 2.50e+00, 1.50e+00
),
cuthoe = cms.vdouble(
2.15e-01, 7.75e-02, 1.47e-01, 3.71e-01, 5.34e-02, 1.23e-01, 5.20e-01, 4.42e-01, 4.04e-01
),
cuthoel = cms.vdouble(
2.36e-01, 8.36e-02, 1.45e-01, 3.75e-01, 3.92e-02, 9.79e-02, 3.63e-01, 3.83e-01, 3.39e-01
),
cutip_gsf = cms.vdouble(
1.98e-02, 7.41e-02, 9.79e-02, 6.49e-02, 5.50e-01, 2.04e-01, 9.13e-02, 1.95e-01, 9.38e-02
),
cutip_gsfl = cms.vdouble(
1.54e-02, 7.26e-02, 5.62e-02, 2.12e-02, 6.20e-01, 2.67e-01, 1.09e-01, 1.60e-01, 4.79e-02
),
cutiso_sum = cms.vdouble(
2.15e+01, 1.44e+01, 1.45e+01, 1.46e+01, 8.20e+00, 1.08e+01, 8.66e+00, 1.57e+01, 2.98e+00
),
cutiso_sumoet = cms.vdouble(
1.79e+01, 9.63e+00, 7.85e+00, 1.23e+01, 5.08e+00, 6.50e+00, 1.10e+01, 8.84e+00, 7.70e+00
),
cutiso_sumoetl = cms.vdouble(
8.08e+00, 7.97e+00, 6.49e+00, 7.86e+00, 4.03e+00, 5.32e+00, 7.20e+00, 7.89e+00, 3.82e+00
),
cutsee = cms.vdouble(
1.45e-02, 1.16e-02, 1.54e-02, 3.90e-02, 3.28e-02, 3.66e-02, 1.45e-02, 6.78e-02, 9.17e-02
),
cutseel = cms.vdouble(
1.32e-02, 1.17e-02, 1.23e-02, 5.23e-02, 3.10e-02, 3.41e-02, 1.27e-02, 5.89e-02, 5.44e-02
)
)

eidTightMC = eidCutBasedExt.clone()
eidTightMC.electronIDType = 'classbased'
eidTightMC.electronQuality = 'tight'
eidTightMC.electronVersion = 'V06'
eidTightMC.additionalCategories = True
eidTightMC.classbasedtightEleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutdetain = cms.vdouble(
1.16e-02, 4.49e-03, 9.38e-03, 1.84e-02, 6.78e-03, 1.09e-02, 2.52e-02, 2.68e-02, 1.39e-02
),
cutdetainl = cms.vdouble(
8.16e-03, 4.01e-03, 8.10e-03, 1.90e-02, 5.88e-03, 8.93e-03, 1.71e-02, 4.34e-02, 1.43e-02
),
cutdphiin = cms.vdouble(
8.97e-02, 9.93e-02, 2.95e-01, 9.79e-02, 1.51e-01, 2.52e-01, 3.41e-01, 3.08e-01, 3.28e-01
),
cutdphiinl = cms.vdouble(
6.10e-02, 1.40e-01, 2.86e-01, 9.21e-02, 1.97e-01, 2.40e-01, 3.33e-01, 3.03e-01, 2.58e-01
),
cuteseedopcor = cms.vdouble(
6.37e-01, 9.43e-01, 7.42e-01, 7.48e-01, 7.63e-01, 6.31e-01, 2.14e-01, 8.73e-01, 4.73e-01
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 1.50e+00, 2.50e+00, 2.50e+00, 1.50e+00, 1.50e+00, 2.50e+00, 5.00e-01
),
cuthoe = cms.vdouble(
2.15e-01, 6.08e-02, 1.47e-01, 3.69e-01, 3.49e-02, 1.02e-01, 5.20e-01, 4.22e-01, 4.04e-01
),
cuthoel = cms.vdouble(
2.28e-01, 8.36e-02, 1.43e-01, 3.70e-01, 3.92e-02, 9.79e-02, 3.00e-01, 3.81e-01, 3.39e-01
),
cutip_gsf = cms.vdouble(
1.31e-02, 5.86e-02, 8.39e-02, 3.66e-02, 4.52e-01, 2.04e-01, 9.13e-02, 8.02e-02, 7.31e-02
),
cutip_gsfl = cms.vdouble(
1.19e-02, 5.27e-02, 4.71e-02, 2.12e-02, 2.33e-01, 2.67e-01, 1.09e-01, 1.22e-01, 4.79e-02
),
cutiso_sum = cms.vdouble(
1.55e+01, 1.22e+01, 1.22e+01, 1.17e+01, 7.16e+00, 9.71e+00, 8.66e+00, 1.19e+01, 2.98e+00
),
cutiso_sumoet = cms.vdouble(
1.19e+01, 7.81e+00, 6.28e+00, 8.92e+00, 4.65e+00, 5.49e+00, 9.36e+00, 8.84e+00, 5.94e+00
),
cutiso_sumoetl = cms.vdouble(
6.21e+00, 6.81e+00, 5.30e+00, 5.39e+00, 2.73e+00, 4.73e+00, 4.84e+00, 3.46e+00, 3.73e+00
),
cutsee = cms.vdouble(
1.45e-02, 1.16e-02, 1.20e-02, 3.90e-02, 2.97e-02, 3.11e-02, 9.87e-03, 3.47e-02, 9.17e-02
),
cutseel = cms.vdouble(
1.32e-02, 1.17e-02, 1.12e-02, 3.87e-02, 2.81e-02, 2.87e-02, 9.87e-03, 2.96e-02, 5.44e-02
)
)

eidSuperTightMC = eidCutBasedExt.clone()
eidSuperTightMC.electronIDType = 'classbased'
eidSuperTightMC.electronQuality = 'supertight'
eidSuperTightMC.electronVersion = 'V06'
eidSuperTightMC.additionalCategories = True
eidSuperTightMC.classbasedsupertightEleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutdetain = cms.vdouble(
1.09e-02, 3.48e-03, 6.80e-03, 1.84e-02, 6.78e-03, 9.08e-03, 8.59e-03, 1.46e-02, 1.39e-02
),
cutdetainl = cms.vdouble(
7.84e-03, 3.55e-03, 6.92e-03, 1.90e-02, 5.88e-03, 8.93e-03, 1.71e-02, 4.34e-02, 1.43e-02
),
cutdphiin = cms.vdouble(
8.97e-02, 6.73e-02, 2.27e-01, 8.25e-02, 1.20e-01, 2.31e-01, 1.82e-01, 2.89e-01, 3.28e-01
),
cutdphiinl = cms.vdouble(
6.10e-02, 3.14e-02, 2.86e-01, 8.16e-02, 7.02e-02, 2.40e-01, 3.33e-01, 3.03e-01, 2.58e-01
),
cuteseedopcor = cms.vdouble(
6.42e-01, 9.51e-01, 9.76e-01, 7.97e-01, 7.63e-01, 8.72e-01, 3.16e-01, 8.77e-01, 4.73e-01
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 5.00e-01, 2.50e+00, 1.50e+00, 5.00e-01, 1.50e+00, 1.50e+00, 5.00e-01
),
cuthoe = cms.vdouble(
1.64e-01, 5.06e-02, 1.47e-01, 3.63e-01, 2.35e-02, 1.02e-01, 4.88e-01, 4.20e-01, 4.04e-01
),
cuthoel = cms.vdouble(
3.79e-02, 8.36e-02, 1.43e-01, 3.70e-01, 3.33e-02, 9.79e-02, 2.57e-01, 3.80e-01, 3.39e-01
),
cutip_gsf = cms.vdouble(
1.29e-02, 4.75e-02, 8.39e-02, 1.96e-02, 3.64e-01, 2.04e-01, 9.13e-02, 8.02e-02, 6.53e-02
),
cutip_gsfl = cms.vdouble(
1.19e-02, 4.26e-02, 4.71e-02, 1.12e-02, 2.33e-01, 2.67e-01, 1.09e-01, 1.22e-01, 4.79e-02
),
cutiso_sum = cms.vdouble(
1.30e+01, 7.77e+00, 9.63e+00, 1.02e+01, 4.76e+00, 8.40e+00, 5.36e+00, 9.92e+00, 2.21e+00
),
cutiso_sumoet = cms.vdouble(
9.10e+00, 6.41e+00, 5.54e+00, 7.10e+00, 4.65e+00, 4.56e+00, 8.98e+00, 8.47e+00, 5.94e+00
),
cutiso_sumoetl = cms.vdouble(
5.31e+00, 5.99e+00, 3.82e+00, 4.40e+00, 2.25e+00, 3.81e+00, 3.45e+00, 2.68e+00, 3.20e+00
),
cutsee = cms.vdouble(
1.43e-02, 1.10e-02, 1.17e-02, 3.50e-02, 2.92e-02, 3.00e-02, 9.87e-03, 3.47e-02, 9.17e-02
),
cutseel = cms.vdouble(
1.32e-02, 1.09e-02, 1.09e-02, 3.51e-02, 2.78e-02, 2.75e-02, 9.87e-03, 2.96e-02, 5.44e-02
)
)

eidHyperTight1MC = eidCutBasedExt.clone()
eidHyperTight1MC.electronIDType = 'classbased'
eidHyperTight1MC.electronQuality = 'hypertight1'
eidHyperTight1MC.electronVersion = 'V06'
eidHyperTight1MC.additionalCategories = True
eidHyperTight1MC.classbasedhypertight1EleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutdetain = cms.vdouble(
9.93e-03, 3.48e-03, 6.58e-03, 1.65e-02, 6.78e-03, 6.84e-03, 5.95e-03, 9.41e-03, 1.39e-02
),
cutdetainl = cms.vdouble(
3.75e-03, 3.55e-03, 6.03e-03, 1.64e-02, 5.88e-03, 6.34e-03, 1.06e-02, 4.34e-02, 1.12e-02
),
cutdphiin = cms.vdouble(
6.38e-02, 3.62e-02, 2.27e-01, 7.54e-02, 3.00e-02, 1.16e-01, 1.82e-01, 2.76e-01, 3.28e-01
),
cutdphiinl = cms.vdouble(
4.97e-02, 1.44e-02, 2.86e-01, 4.76e-02, 5.60e-02, 2.26e-01, 3.33e-01, 2.60e-01, 2.58e-01
),
cuteseedopcor = cms.vdouble(
7.22e-01, 9.66e-01, 1.00e+00, 8.42e-01, 7.63e-01, 9.04e-01, 5.18e-01, 9.34e-01, 4.73e-01
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 5.00e-01, 1.50e+00, 5.00e-01, 5.00e-01, 1.50e+00, 1.50e+00, -5.00e-01
),
cuthoe = cms.vdouble(
7.93e-02, 4.07e-02, 1.47e-01, 3.40e-01, 2.14e-02, 1.02e-01, 3.41e-01, 4.18e-01, 4.04e-01
),
cuthoel = cms.vdouble(
3.79e-02, 7.78e-02, 9.91e-02, 3.33e-01, 1.94e-02, 9.79e-02, 2.57e-01, 3.69e-01, 3.39e-01
),
cutip_gsf = cms.vdouble(
1.26e-02, 1.85e-02, 8.39e-02, 1.70e-02, 1.32e-01, 2.00e-01, 8.97e-02, 8.02e-02, 2.66e-02
),
cutip_gsfl = cms.vdouble(
1.19e-02, 2.70e-02, 4.71e-02, 1.10e-02, 1.16e-01, 1.99e-01, 9.02e-02, 1.22e-01, 4.79e-02
),
cutiso_sum = cms.vdouble(
1.02e+01, 7.77e+00, 8.59e+00, 6.93e+00, 4.43e+00, 5.82e+00, 3.97e+00, 5.10e+00, 1.98e+00
),
cutiso_sumoet = cms.vdouble(
7.87e+00, 5.96e+00, 5.35e+00, 6.72e+00, 4.51e+00, 3.98e+00, 8.14e+00, 6.26e+00, 4.80e+00
),
cutiso_sumoetl = cms.vdouble(
4.17e+00, 4.63e+00, 3.08e+00, 3.77e+00, 2.02e+00, 3.68e+00, 3.45e+00, 2.41e+00, 2.58e+00
),
cutsee = cms.vdouble(
1.13e-02, 1.07e-02, 1.17e-02, 3.21e-02, 2.92e-02, 2.91e-02, 9.78e-03, 3.47e-02, 3.96e-02
),
cutseel = cms.vdouble(
1.32e-02, 1.03e-02, 1.03e-02, 3.43e-02, 2.67e-02, 2.69e-02, 9.76e-03, 2.96e-02, 5.44e-02
)
)

eidHyperTight2MC = eidCutBasedExt.clone()
eidHyperTight2MC.electronIDType = 'classbased'
eidHyperTight2MC.electronQuality = 'hypertight2'
eidHyperTight2MC.electronVersion = 'V06'
eidHyperTight2MC.additionalCategories = True
eidHyperTight2MC.classbasedhypertight2EleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutdetain = cms.vdouble(
9.93e-03, 3.48e-03, 6.58e-03, 1.65e-02, 6.78e-03, 6.84e-03, 5.95e-03, 9.41e-03, 1.39e-02
),
cutdetainl = cms.vdouble(
3.75e-03, 3.55e-03, 6.03e-03, 1.64e-02, 5.88e-03, 6.34e-03, 1.06e-02, 4.34e-02, 1.12e-02
),
cutdphiin = cms.vdouble(
6.38e-02, 3.62e-02, 2.27e-01, 7.54e-02, 3.00e-02, 1.16e-01, 1.82e-01, 2.76e-01, 3.28e-01
),
cutdphiinl = cms.vdouble(
4.97e-02, 1.44e-02, 2.86e-01, 4.76e-02, 5.60e-02, 2.26e-01, 3.33e-01, 2.60e-01, 2.58e-01
),
cuteseedopcor = cms.vdouble(
7.22e-01, 9.66e-01, 1.00e+00, 8.42e-01, 7.63e-01, 9.04e-01, 5.18e-01, 9.34e-01, 4.73e-01
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 5.00e-01, 1.50e+00, 5.00e-01, 5.00e-01, 1.50e+00, 1.50e+00, -5.00e-01
),
cuthoe = cms.vdouble(
7.93e-02, 4.07e-02, 1.47e-01, 3.40e-01, 2.14e-02, 1.02e-01, 3.41e-01, 4.18e-01, 4.04e-01
),
cuthoel = cms.vdouble(
3.79e-02, 7.78e-02, 9.91e-02, 3.33e-01, 1.94e-02, 9.79e-02, 2.57e-01, 3.69e-01, 3.39e-01
),
cutip_gsf = cms.vdouble(
1.26e-02, 1.85e-02, 8.39e-02, 1.70e-02, 1.32e-01, 2.00e-01, 8.97e-02, 8.02e-02, 2.66e-02
),
cutip_gsfl = cms.vdouble(
1.19e-02, 2.70e-02, 4.71e-02, 1.10e-02, 1.16e-01, 1.99e-01, 9.02e-02, 1.22e-01, 4.79e-02
),
cutiso_sum = cms.vdouble(
1.02e+01, 7.77e+00, 8.59e+00, 6.93e+00, 4.43e+00, 5.82e+00, 3.97e+00, 5.10e+00, 1.98e+00
),
cutiso_sumoet = cms.vdouble(
7.87e+00, 5.96e+00, 5.35e+00, 6.72e+00, 4.51e+00, 3.98e+00, 8.14e+00, 6.26e+00, 4.80e+00
),
cutiso_sumoetl = cms.vdouble(
4.17e+00, 4.63e+00, 3.08e+00, 3.77e+00, 2.02e+00, 3.68e+00, 3.45e+00, 2.41e+00, 2.58e+00
),
cutsee = cms.vdouble(
1.13e-02, 1.07e-02, 1.17e-02, 3.21e-02, 2.92e-02, 2.91e-02, 9.78e-03, 3.47e-02, 3.96e-02
),
cutseel = cms.vdouble(
1.32e-02, 1.03e-02, 1.03e-02, 3.43e-02, 2.67e-02, 2.69e-02, 9.76e-03, 2.96e-02, 5.44e-02
)
)

eidHyperTight3MC = eidCutBasedExt.clone()
eidHyperTight3MC.electronIDType = 'classbased'
eidHyperTight3MC.electronQuality = 'hypertight3'
eidHyperTight3MC.electronVersion = 'V06'
eidHyperTight3MC.additionalCategories = True
eidHyperTight3MC.classbasedhypertight3EleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutdetain = cms.vdouble(
9.93e-03, 3.48e-03, 6.58e-03, 1.65e-02, 6.78e-03, 6.84e-03, 5.95e-03, 9.41e-03, 1.39e-02
),
cutdetainl = cms.vdouble(
3.75e-03, 3.55e-03, 6.03e-03, 1.64e-02, 5.88e-03, 6.34e-03, 1.06e-02, 4.34e-02, 1.12e-02
),
cutdphiin = cms.vdouble(
6.38e-02, 3.62e-02, 2.27e-01, 7.54e-02, 3.00e-02, 1.16e-01, 1.82e-01, 2.76e-01, 3.28e-01
),
cutdphiinl = cms.vdouble(
4.97e-02, 1.44e-02, 2.86e-01, 4.76e-02, 5.60e-02, 2.26e-01, 3.33e-01, 2.60e-01, 2.58e-01
),
cuteseedopcor = cms.vdouble(
7.22e-01, 9.66e-01, 1.00e+00, 8.42e-01, 7.63e-01, 9.04e-01, 5.18e-01, 9.34e-01, 4.73e-01
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 5.00e-01, 1.50e+00, 5.00e-01, 5.00e-01, 1.50e+00, 1.50e+00, -5.00e-01
),
cuthoe = cms.vdouble(
7.93e-02, 4.07e-02, 1.47e-01, 3.40e-01, 2.14e-02, 1.02e-01, 3.41e-01, 4.18e-01, 4.04e-01
),
cuthoel = cms.vdouble(
3.79e-02, 7.78e-02, 9.91e-02, 3.33e-01, 1.94e-02, 9.79e-02, 2.57e-01, 3.69e-01, 3.39e-01
),
cutip_gsf = cms.vdouble(
1.26e-02, 1.85e-02, 8.39e-02, 1.70e-02, 1.32e-01, 2.00e-01, 8.97e-02, 8.02e-02, 2.66e-02
),
cutip_gsfl = cms.vdouble(
1.19e-02, 2.70e-02, 4.71e-02, 1.10e-02, 1.16e-01, 1.99e-01, 9.02e-02, 1.22e-01, 4.79e-02
),
cutiso_sum = cms.vdouble(
1.02e+01, 7.77e+00, 8.59e+00, 6.93e+00, 4.43e+00, 5.82e+00, 3.97e+00, 5.10e+00, 1.98e+00
),
cutiso_sumoet = cms.vdouble(
7.87e+00, 5.96e+00, 5.35e+00, 6.72e+00, 4.51e+00, 3.98e+00, 8.14e+00, 6.26e+00, 4.80e+00
),
cutiso_sumoetl = cms.vdouble(
4.17e+00, 4.63e+00, 3.08e+00, 3.77e+00, 2.02e+00, 3.68e+00, 3.45e+00, 2.41e+00, 2.58e+00
),
cutsee = cms.vdouble(
1.13e-02, 1.07e-02, 1.17e-02, 3.21e-02, 2.92e-02, 2.91e-02, 9.78e-03, 3.47e-02, 3.96e-02
),
cutseel = cms.vdouble(
1.32e-02, 1.03e-02, 1.03e-02, 3.43e-02, 2.67e-02, 2.69e-02, 9.76e-03, 2.96e-02, 5.44e-02
)
)

eidHyperTight4MC = eidCutBasedExt.clone()
eidHyperTight4MC.electronIDType = 'classbased'
eidHyperTight4MC.electronQuality = 'hypertight4'
eidHyperTight4MC.electronVersion = 'V06'
eidHyperTight4MC.additionalCategories = True
eidHyperTight4MC.classbasedhypertight4EleIDCutsV06 = cms.PSet(
cutdcotdist = cms.vdouble(9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.),
cutdetain = cms.vdouble(
9.93e-03, 3.48e-03, 6.58e-03, 1.65e-02, 6.78e-03, 6.84e-03, 5.95e-03, 9.41e-03, 1.39e-02
),
cutdetainl = cms.vdouble(
3.75e-03, 3.55e-03, 6.03e-03, 1.64e-02, 5.88e-03, 6.34e-03, 1.06e-02, 4.34e-02, 1.12e-02
),
cutdphiin = cms.vdouble(
6.38e-02, 3.62e-02, 2.27e-01, 7.54e-02, 3.00e-02, 1.16e-01, 1.82e-01, 2.76e-01, 3.28e-01
),
cutdphiinl = cms.vdouble(
4.97e-02, 1.44e-02, 2.86e-01, 4.76e-02, 5.60e-02, 2.26e-01, 3.33e-01, 2.60e-01, 2.58e-01
),
cuteseedopcor = cms.vdouble(
7.22e-01, 9.66e-01, 1.00e+00, 8.42e-01, 7.63e-01, 9.04e-01, 5.18e-01, 9.34e-01, 4.73e-01
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 5.00e-01, 1.50e+00, 5.00e-01, 5.00e-01, 1.50e+00, 1.50e+00, -5.00e-01
),
cuthoe = cms.vdouble(
7.93e-02, 4.07e-02, 1.47e-01, 3.40e-01, 2.14e-02, 1.02e-01, 3.41e-01, 4.18e-01, 4.04e-01
),
cuthoel = cms.vdouble(
3.79e-02, 7.78e-02, 9.91e-02, 3.33e-01, 1.94e-02, 9.79e-02, 2.57e-01, 3.69e-01, 3.39e-01
),
cutip_gsf = cms.vdouble(
1.26e-02, 1.85e-02, 8.39e-02, 1.70e-02, 1.32e-01, 2.00e-01, 8.97e-02, 8.02e-02, 2.66e-02
),
cutip_gsfl = cms.vdouble(
1.19e-02, 2.70e-02, 4.71e-02, 1.10e-02, 1.16e-01, 1.99e-01, 9.02e-02, 1.22e-01, 4.79e-02
),
cutiso_sum = cms.vdouble(
1.02e+01, 7.77e+00, 8.59e+00, 6.93e+00, 4.43e+00, 5.82e+00, 3.97e+00, 5.10e+00, 1.98e+00
),
cutiso_sumoet = cms.vdouble(
7.87e+00, 5.96e+00, 5.35e+00, 6.72e+00, 4.51e+00, 3.98e+00, 8.14e+00, 6.26e+00, 4.80e+00
),
cutiso_sumoetl = cms.vdouble(
4.17e+00, 4.63e+00, 3.08e+00, 3.77e+00, 2.02e+00, 3.68e+00, 3.45e+00, 2.41e+00, 2.58e+00
),
cutsee = cms.vdouble(
1.13e-02, 1.07e-02, 1.17e-02, 3.21e-02, 2.92e-02, 2.91e-02, 9.78e-03, 3.47e-02, 3.96e-02
),
cutseel = cms.vdouble(
1.32e-02, 1.03e-02, 1.03e-02, 3.43e-02, 2.67e-02, 2.69e-02, 9.76e-03, 2.96e-02, 5.44e-02
)
)
