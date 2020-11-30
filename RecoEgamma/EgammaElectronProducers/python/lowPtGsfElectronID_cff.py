import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.defaultLowPtGsfElectronID_cfi import defaultLowPtGsfElectronID

lowPtGsfElectronID = defaultLowPtGsfElectronID.clone(
    ModelNames = cms.vstring(['']),
    ModelWeights = cms.vstring([
            'RecoEgamma/ElectronIdentification/data/LowPtElectrons/RunII_Autumn18_LowPtElectrons_mva_id.xml.gz',
            ]),
    ModelThresholds = cms.vdouble([-10.])
    )

from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL
from Configuration.Eras.Modifier_run2_miniAOD_devel_cff import run2_miniAOD_devel
from Configuration.Eras.Modifier_bParking_cff import bParking
run2_miniAOD_UL.toModify(
    lowPtGsfElectronID,
    rho = "fixedGridRhoFastjetAll",
    ModelWeights = ["RecoEgamma/ElectronIdentification/data/LowPtElectrons/LowPtElectrons_ID_2020Sept15.root"],
    ModelThresholds = [-99.],
    Version = "V1",
)
run2_miniAOD_devel.toModify(
    lowPtGsfElectronID,
    ModelWeights = ["RecoEgamma/ElectronIdentification/data/LowPtElectrons/LowPtElectrons_ID_2020Nov28.root"],
)
(bParking & run2_miniAOD_UL).toModify(
    lowPtGsfElectronID,
    ModelWeights = ["RecoEgamma/ElectronIdentification/data/LowPtElectrons/LowPtElectrons_ID_2021May17.root"],
)
