import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi import *

eidVeryLoose = eidCutBasedExt.clone()
eidVeryLoose.electronIDType = 'classbased'
eidVeryLoose.electronQuality = 'veryloose'
eidVeryLoose.electronVersion = 'V04'
eidVeryLoose.etBinning = False
eidVeryLoose.additionalCategories = False
eidVeryLoose.classbasedverylooseEleIDCutsV04 = cms.PSet(
cutdcotdist = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutet = cms.vdouble(
0., 0., 0., 0., 0., 0., 0., 0., 0.
),
cutip_gsf = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sum = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sumoet = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutdetain = cms.vdouble(
1.30e-02, 1.00e-02, 3.10e-02, 2.80e-02, 1.26e-02, 2.89e-02, 2.42e-02, 5.13e-02, 2.21e-02
),
cutdphiin = cms.vdouble(
7.51e-02, 3.71e-01, 4.20e-01, 9.86e-02, 2.88e-01, 3.89e-01, 3.77e-01, 4.32e-01, 4.57e-01
),
cuteseedopcor = cms.vdouble(
6.31e-01, 2.37e-01, 3.04e-01, 8.04e-01, 1.63e-01, 5.03e-01, 2.78e-01, 3.10e-01, 1.31e-01
),
cutfmishits = cms.vdouble(
4.50e+00, 1.50e+00, 1.50e+00, 7.50e+00, 2.50e+00, 2.50e+00, 3.50e+00, 4.50e+00, 3.50e+00
),
cuthoe = cms.vdouble(
2.47e-01, 1.11e-01, 1.49e-01, 3.82e-01, 7.17e-02, 1.47e-01, 1.16e+00, 5.04e+00, 3.33e+00
),
cutsee = cms.vdouble(
1.92e-02, 1.98e-02, 2.53e-02, 5.28e-02, 3.91e-02, 4.61e-02, 2.66e-02, 6.58e-02, 3.20e+00
)
)

eidLoose = eidCutBasedExt.clone()
eidLoose.electronIDType = 'classbased'
eidLoose.electronQuality = 'loose'
eidLoose.electronVersion = 'V04'
eidLoose.etBinning = False
eidLoose.additionalCategories = False
eidLoose.classbasedlooseEleIDCutsV04 = cms.PSet(
cutdcotdist = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutet = cms.vdouble(
0., 0., 0., 0., 0., 0., 0., 0., 0.
),
cutip_gsf = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sum = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sumoet = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutdetain = cms.vdouble(
1.30e-02, 5.95e-03, 3.10e-02, 1.68e-02, 8.44e-03, 1.70e-02, 1.55e-02, 5.13e-02, 1.61e-02
),
cutdphiin = cms.vdouble(
7.51e-02, 3.30e-01, 4.20e-01, 9.86e-02, 2.84e-01, 3.28e-01, 3.77e-01, 4.32e-01, 3.74e-01
),
cuteseedopcor = cms.vdouble(
6.31e-01, 3.02e-01, 3.04e-01, 8.10e-01, 2.23e-01, 5.03e-01, 2.78e-01, 3.10e-01, 4.69e-01
),
cutfmishits = cms.vdouble(
4.50e+00, 1.50e+00, 1.50e+00, 2.50e+00, 2.50e+00, 1.50e+00, 2.50e+00, 4.50e+00, 5.00e-01
),
cuthoe = cms.vdouble(
2.47e-01, 7.78e-02, 1.49e-01, 3.82e-01, 4.70e-02, 1.12e-01, 1.16e+00, 5.04e+00, 1.35e+00
),
cutsee = cms.vdouble(
1.92e-02, 1.31e-02, 2.53e-02, 5.27e-02, 3.29e-02, 4.19e-02, 2.65e-02, 6.58e-02, 1.38e-01
)
)

eidMedium = eidCutBasedExt.clone()
eidMedium.electronIDType = 'classbased'
eidMedium.electronQuality = 'medium'
eidMedium.electronVersion = 'V04'
eidMedium.etBinning = False
eidMedium.additionalCategories = False
eidMedium.classbasedmediumEleIDCutsV04 = cms.PSet(
cutdcotdist = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutet = cms.vdouble(
0., 0., 0., 0., 0., 0., 0., 0., 0.
),
cutip_gsf = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sum = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sumoet = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutdetain = cms.vdouble(
1.19e-02, 4.20e-03, 1.07e-02, 1.49e-02, 6.56e-03, 1.19e-02, 1.16e-02, 5.13e-02, 6.37e-03
),
cutdphiin = cms.vdouble(
7.51e-02, 2.93e-01, 3.58e-01, 9.53e-02, 1.62e-01, 2.99e-01, 2.76e-01, 4.32e-01, 2.57e-01
),
cuteseedopcor = cms.vdouble(
6.31e-01, 8.14e-01, 7.60e-01, 8.18e-01, 7.56e-01, 5.35e-01, 6.20e-01, 7.88e-01, 8.85e-01
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 2.50e+00, 1.50e+00, 5.00e-01
),
cuthoe = cms.vdouble(
2.46e-01, 6.80e-02, 1.35e-01, 3.73e-01, 2.33e-02, 5.58e-02, 8.80e-01, 5.04e+00, 3.78e-02
),
cutsee = cms.vdouble(
1.92e-02, 1.13e-02, 1.47e-02, 3.84e-02, 3.05e-02, 3.36e-02, 1.35e-02, 5.05e-02, 2.79e-02
)
)

eidTight = eidCutBasedExt.clone()
eidTight.electronIDType = 'classbased'
eidTight.electronQuality = 'tight'
eidTight.electronVersion = 'V04'
eidTight.etBinning = False
eidTight.additionalCategories = False
eidTight.classbasedtightEleIDCutsV04 = cms.PSet(
cutdcotdist = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutet = cms.vdouble(
0., 0., 0., 0., 0., 0., 0., 0., 0.
),
cutip_gsf = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sum = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sumoet = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutdetain = cms.vdouble(
9.28e-03, 3.56e-03, 7.16e-03, 1.31e-02, 5.81e-03, 9.79e-03, 1.15e-02, 1.66e-02, 3.19e-03
),
cutdphiin = cms.vdouble(
4.66e-02, 7.80e-02, 2.64e-01, 4.42e-02, 3.20e-02, 2.37e-01, 8.25e-02, 2.07e-01, 5.39e-02
),
cuteseedopcor = cms.vdouble(
6.48e-01, 8.97e-01, 8.91e-01, 8.39e-01, 8.35e-01, 6.49e-01, 6.76e-01, 8.70e-01, 9.91e-01
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 1.50e+00, 5.00e-01, 2.50e+00, 5.00e-01, 5.00e-01
),
cuthoe = cms.vdouble(
9.94e-02, 5.61e-02, 1.05e-01, 9.73e-02, 1.81e-02, 3.06e-02, 5.57e-01, 5.04e+00, 1.06e-03
),
cutsee = cms.vdouble(
1.56e-02, 1.07e-02, 1.23e-02, 3.35e-02, 2.98e-02, 3.06e-02, 1.07e-02, 3.79e-02, 1.01e-02
)
)

eidSuperTight = eidCutBasedExt.clone()
eidSuperTight.electronIDType = 'classbased'
eidSuperTight.electronQuality = 'supertight'
eidSuperTight.electronVersion = 'V04'
eidSuperTight.etBinning = False
eidSuperTight.additionalCategories = False
eidSuperTight.classbasedsupertightEleIDCutsV04 = cms.PSet(
cutdcotdist = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutet = cms.vdouble(
0., 0., 0., 0., 0., 0., 0., 0., 0.
),
cutip_gsf = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sum = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sumoet = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutdetain = cms.vdouble(
9.28e-03, 3.41e-03, 5.60e-03, 9.00e-03, 5.15e-03, 8.03e-03, 1.06e-02, 1.51e-02, 3.15e-03
),
cutdphiin = cms.vdouble(
3.23e-02, 3.51e-02, 1.61e-01, 3.06e-02, 2.07e-02, 6.15e-02, 5.82e-02, 6.05e-02, 3.66e-02
),
cuteseedopcor = cms.vdouble(
7.35e-01, 9.41e-01, 9.53e-01, 8.86e-01, 8.85e-01, 9.38e-01, 7.98e-01, 9.26e-01, 1.02e+00
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 5.00e-01, 1.50e+00, 5.00e-01, 5.00e-01, 2.50e+00, 5.00e-01, 5.00e-01
),
cuthoe = cms.vdouble(
5.25e-02, 4.62e-02, 4.98e-02, 5.05e-02, 1.60e-02, 2.04e-02, 2.18e-01, 5.02e+00, 2.96e-05
),
cutsee = cms.vdouble(
1.22e-02, 1.04e-02, 1.12e-02, 3.01e-02, 2.82e-02, 2.88e-02, 9.95e-03, 2.74e-02, 9.07e-03
)
)

eidHyperTight1 = eidCutBasedExt.clone()
eidHyperTight1.electronIDType = 'classbased'
eidHyperTight1.electronQuality = 'hypertight1'
eidHyperTight1.electronVersion = 'V04'
eidHyperTight1.etBinning = False
eidHyperTight1.additionalCategories = False
eidHyperTight1.classbasedhypertight1EleIDCutsV04 = cms.PSet(
cutdcotdist = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutet = cms.vdouble(
0., 0., 0., 0., 0., 0., 0., 0., 0.
),
cutip_gsf = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sum = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sumoet = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutdetain = cms.vdouble(
6.70e-03, 3.18e-03, 4.15e-03, 8.05e-03, 3.95e-03, 6.92e-03, 1.06e-02, 1.46e-02, 1.15e-03
),
cutdphiin = cms.vdouble(
2.07e-02, 2.22e-02, 9.49e-02, 2.43e-02, 1.58e-02, 2.70e-02, 3.84e-02, 3.84e-02, 3.47e-02
),
cuteseedopcor = cms.vdouble(
1.04e+00, 9.63e-01, 9.91e-01, 9.69e-01, 8.98e-01, 9.55e-01, 8.16e-01, 9.76e-01, 1.04e+00
),
cutfmishits = cms.vdouble(
1.50e+00, 1.50e+00, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, -5.00e-01
),
cuthoe = cms.vdouble(
3.83e-02, 3.84e-02, 3.50e-02, 2.33e-02, 1.26e-02, 1.64e-02, 1.12e-01, 2.72e+00, 2.30e-06
),
cutsee = cms.vdouble(
1.05e-02, 1.01e-02, 1.02e-02, 2.86e-02, 2.72e-02, 2.75e-02, 9.67e-03, 2.54e-02, 8.85e-03
)
)

eidHyperTight2 = eidCutBasedExt.clone()
eidHyperTight2.electronIDType = 'classbased'
eidHyperTight2.electronQuality = 'hypertight2'
eidHyperTight2.electronVersion = 'V04'
eidHyperTight2.etBinning = False
eidHyperTight2.additionalCategories = False
eidHyperTight2.classbasedhypertight2EleIDCutsV04 = cms.PSet(
cutdcotdist = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutet = cms.vdouble(
0., 0., 0., 0., 0., 0., 0., 0., 0.
),
cutip_gsf = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sum = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sumoet = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutdetain = cms.vdouble(
3.12e-03, 2.09e-03, 3.22e-03, 4.93e-03, 2.99e-03, 5.23e-03, 8.85e-03, 1.21e-02, 1.15e-03
),
cutdphiin = cms.vdouble(
1.45e-02, 7.75e-03, 3.30e-02, 1.17e-02, 7.93e-03, 1.64e-02, 2.10e-02, 2.40e-02, 3.47e-02
),
cuteseedopcor = cms.vdouble(
1.09e+00, 9.90e-01, 1.07e+00, 9.71e-01, 9.60e-01, 9.55e-01, 9.87e-01, 1.04e+00, 1.04e+00
),
cutfmishits = cms.vdouble(
5.00e-01, 1.50e+00, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, -5.00e-01
),
cuthoe = cms.vdouble(
3.25e-02, 3.29e-02, 2.96e-02, 1.52e-02, 1.22e-02, 1.31e-02, 6.02e-02, 1.22e-01, 2.30e-06
),
cutsee = cms.vdouble(
9.85e-03, 9.79e-03, 9.64e-03, 2.72e-02, 2.64e-02, 2.65e-02, 9.37e-03, 2.37e-02, 8.85e-03
)
)

eidHyperTight3 = eidCutBasedExt.clone()
eidHyperTight3.electronIDType = 'classbased'
eidHyperTight3.electronQuality = 'hypertight3'
eidHyperTight3.electronVersion = 'V04'
eidHyperTight3.etBinning = False
eidHyperTight3.additionalCategories = False
eidHyperTight3.classbasedhypertight3EleIDCutsV04 = cms.PSet(
cutdcotdist = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutet = cms.vdouble(
0., 0., 0., 0., 0., 0., 0., 0., 0.
),
cutip_gsf = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sum = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sumoet = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutdetain = cms.vdouble(
1.73e-03, 1.23e-03, 2.47e-03, 4.18e-03, 2.10e-03, 3.28e-03, 8.85e-03, 7.63e-03, 1.15e-03
),
cutdphiin = cms.vdouble(
5.32e-03, 2.99e-03, 1.72e-02, 6.27e-03, 7.93e-03, 9.69e-03, 7.60e-03, 1.67e-02, 3.47e-02
),
cuteseedopcor = cms.vdouble(
1.14e+00, 1.00e+00, 1.14e+00, 1.02e+00, 9.64e-01, 1.15e+00, 1.05e+00, 1.22e+00, 1.04e+00
),
cutfmishits = cms.vdouble(
5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, -5.00e-01, 5.00e-01, 5.00e-01, 5.00e-01, -5.00e-01
),
cuthoe = cms.vdouble(
2.93e-02, 2.80e-02, 2.81e-02, 1.35e-02, 9.55e-03, 1.21e-02, 3.27e-02, 3.45e-02, 2.30e-06
),
cutsee = cms.vdouble(
9.41e-03, 9.54e-03, 9.34e-03, 2.62e-02, 2.56e-02, 2.56e-02, 9.08e-03, 2.35e-02, 8.85e-03
)
)

eidHyperTight4 = eidCutBasedExt.clone()
eidHyperTight4.electronIDType = 'classbased'
eidHyperTight4.electronQuality = 'hypertight4'
eidHyperTight4.electronVersion = 'V04'
eidHyperTight4.etBinning = False
eidHyperTight4.additionalCategories = False
eidHyperTight4.classbasedhypertight4EleIDCutsV04 = cms.PSet(
cutdcotdist = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutet = cms.vdouble(
0., 0., 0., 0., 0., 0., 0., 0., 0.
),
cutip_gsf = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sum = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutiso_sumoet = cms.vdouble(
9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.
),
cutdetain = cms.vdouble(
1.47e-03, 9.67e-04, 1.87e-03, 2.27e-03, 2.10e-03, 1.77e-03, 5.30e-03, 3.60e-03, 1.15e-03
),
cutdphiin = cms.vdouble(
5.03e-03, 2.96e-03, 9.84e-03, 5.85e-03, 7.93e-03, 9.08e-03, 7.10e-03, 1.24e-02, 3.47e-02
),
cuteseedopcor = cms.vdouble(
1.15e+00, 1.01e+00, 1.15e+00, 1.13e+00, 9.64e-01, 1.30e+00, 1.07e+00, 1.29e+00, 1.04e+00
),
cutfmishits = cms.vdouble(
-5.00e-01, -5.00e-01, 5.00e-01, -5.00e-01, -5.00e-01, -5.00e-01, -5.00e-01, 5.00e-01, -5.00e-01
),
cuthoe = cms.vdouble(
2.28e-03, 2.18e-03, 2.64e-02, 1.05e-02, 9.55e-03, 7.95e-03, 3.27e-02, 2.63e-02, 2.30e-06
),
cutsee = cms.vdouble(
9.15e-03, 9.35e-03, 9.16e-03, 2.54e-02, 2.56e-02, 2.49e-02, 8.94e-03, 2.33e-02, 8.85e-03
)
)

