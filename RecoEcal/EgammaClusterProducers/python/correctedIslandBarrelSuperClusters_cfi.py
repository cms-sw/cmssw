import FWCore.ParameterSet.Config as cms

# Energy scale correction for Island SuperClusters
correctedIslandBarrelSuperClusters = cms.EDProducer("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.03),
    superClusterAlgo = cms.string('Island'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.InputTag("islandSuperClusters","islandBarrelSuperClusters"),
    applyEnergyCorrection = cms.bool(True),
    isl_fCorrPset = cms.PSet(
        brLinearLowThr = cms.double(0.0),
        fBremVec = cms.vdouble(0.0),
        brLinearHighThr = cms.double(0.0),
        fEtEtaVec = cms.vdouble(0.0)
    ),
    VerbosityLevel = cms.string('ERROR'),
    recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017

from RecoHI.HiEgammaAlgos.HiCorrectedIslandBarrelSuperClusters_cfi import correctedIslandBarrelSuperClusters as _hiCorrectedIslandBarrelSuperClusters

for e in [pA_2016, peripheralPbPb, pp_on_AA, pp_on_XeXe_2017, ppRef_2017]:
    e.toReplaceWith(correctedIslandBarrelSuperClusters, _hiCorrectedIslandBarrelSuperClusters)
