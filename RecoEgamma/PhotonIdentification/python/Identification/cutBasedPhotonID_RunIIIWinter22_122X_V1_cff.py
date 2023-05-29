from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

import FWCore.ParameterSet.Config as cms

# Common functions and classes for ID definition are imported here:
from RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_tools \
    import ( WorkingPoint_V3,
             IsolationCutInputs,
             ClusterIsolationCutInputs,
             HoverECutInputs,
             configureVIDCutBasedPhoID_V6 )
#
# Details of the ID values can be found in the following link
# https://indico.cern.ch/event/1204277/#5-update-on-run3-photon-cut-ba
#

#
# First, define cut values
#


# Loose working point Barrel and Endcap
idName = "cutBasedPhotonID-RunIIIWinter22-122X-V1-loose"
WP_Loose_EB = WorkingPoint_V3(
    idName     ,  # idName
    0.011452   ,  # full5x5_SigmaIEtaIEtaCut
    0.12999    ,  # hOverEWithEACut
# Isolation cuts are generally absIso < C1 + pt*C2, except for HCalClus is < C1 + pt*C2 + pt*pt*C3
    1.8852     ,  # absPFChgHadIsoWithEACut_C1
    0.0        ,  # absPFChgHadIsoWithEACut_C2
    0.70379    ,  # absPFECalClusIsoWithEACut_C1
    0.00065204 ,  # absPFECalClusIsoWithEACut_C2
    6.3440     ,  # absPFHCalClusIsoWithEACut_C1
    0.010055   ,  # absPFHCalClusIsoWithEACut_C2
    0.00005783    # absPFHCalClusIsoWithEACut_C3
    )
WP_Loose_EE = WorkingPoint_V3(
    idName     ,  # idName
    0.027674   ,  # full5x5_SigmaIEtaIEtaCut
    0.15343    ,  # hOverEWithEACut
# Isolation cuts are generally absIso < C1 + pt*C2, except for HCalClus is < C1 + pt*C2 + pt*pt*C3
    1.6540     ,  # absPFChgHadIsoWithEACut_C1
    0.0        ,  # absPFChgHadIsoWithEACut_C2
    6.61585    ,  # absPFECalClusIsoWithEACut_C1
    0.00019549 ,  # absPFECalClusIsoWithEACut_C2
    1.8588     ,  # absPFHCalClusIsoWithEACut_C1
    0.01170    ,  # absPFHCalClusIsoWithEACut_C2
    0.00007476    # absPFHCalClusIsoWithEACut_C3
    )



# Medium working point Barrel and Endcap
idName = "cutBasedPhotonID-RunIIIWinter22-122X-V1-medium"
WP_Medium_EB = WorkingPoint_V3(
    idName     ,  # idName
    0.01001    ,  # full5x5_SigmaIEtaIEtaCut
    0.058305   ,  # hOverEWithEACut
# Isolation cuts are generally absIso < C1 + pt*C2, except for HCalClus is < C1 + pt*C2 + pt*pt*C3
    0.93929    ,  # absPFChgHadIsoWithEACut_C1
    0.0        ,  # absPFChgHadIsoWithEACut_C2
    0.22770    ,  # absPFECalClusIsoWithEACut_C1
    0.00065204 ,  # absPFECalClusIsoWithEACut_C2
    2.1890     ,  # absPFHCalClusIsoWithEACut_C1
    0.010055   ,  # absPFHCalClusIsoWithEACut_C2
    0.00005783    # absPFHCalClusIsoWithEACut_C3
    )

WP_Medium_EE = WorkingPoint_V3(
    idName     ,  #idName
    0.02687    ,  # full5x5_SigmaIEtaIEtaCut
    0.005181   ,  # hOverECutWithEA
# Isolation cuts are generally absIso < C1 + pt*C2, except for HCalClus is < C1 + pt*C2 + pt*pt*C3
    0.97029    ,  # absPFChgHadIsoWithEACut_C1
    0.0        ,  # absPFChaHadIsoWithEACut_C2
    1.124      ,  # absPFECalClusIsoWithEACut_C1
    0.00019549 ,  # absPFECalClusIsoWithEACut_C2
    0.033670   ,  # absPFHCalClusIsowithEACut_C1
    0.01170    ,  # absPFHCalClusIsoWithEACut_C2
    0.00007476    # absPFHCalClusIsoWithEACut_C3
    )



# Tight working point Barrel and Endcap
idName = "cutBasedPhotonID-RunIIIWinter22-122X-V1-tight"
WP_Tight_EB = WorkingPoint_V3(
    idName     ,  # idName
    0.009993   ,  # full5x5_SigmaIEtaIEtaCut
    0.0417588  ,  # hOverECutWithEA
# Isolation cuts are generally absIso < C1 + pt*C2, except for HCalClus is < C1 + pt*C2 + pt*pt*C3
    0.31631    ,  # absPFChgHadIsoWithEACut_C1
    0.0        ,  # absPFChgHadIsoWithEACut_C2
    0.14189    ,  # absPFECalClusIsoWithEACut_C1
    0.00065204 ,  # absPFECalClusIsoWithEACut_C2
    0.39057    ,  # absPFHCalClusIsoWithEACut_C1
    0.0100547  ,  # absPFHCalClusIsoWithEACut_C2
    0.00005783    # absPFHCalClusIsoWithEACut_C3
    )

WP_Tight_EE = WorkingPoint_V3(
    idName     ,  # idName
    0.02687    ,  # full5x5_SigmaIEtaIEtaCut
    0.0025426  ,  # hOverECutWithEA
# Isolation cuts are generally absIso < C1 + pt*C2, except for HCalClus is < C1 + pt*C2 + pt*pt*C3
    0.29266    ,  # absPFChgHadIsoWithEACut_C1
    0.0        ,  # absPFChgHadIsoWithEACut_C2
    1.04269    ,  # absPFECalClusIsoWithEACut_C1
    0.00019549 ,  # absPFECalClusIsoWithEACut_C2
    0.029262   ,  # absPFHCalClusIsowithEACut_C1
    0.01170    ,  # absPFHCalClusIsoWithEACut_C2
    0.00007476    # absPFHCalClusIsoWithEACut_C3
    )


# Second, define where to find the precomputed isolations and what effective
# areas to use for pile-up correction
isoInputs = IsolationCutInputs(
    # chHadIsolationMapName
    'photonIDValueMapProducer:phoChargedIsolation' ,
    # chHadIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/RunIII_Winter22/effectiveArea_ChgHadronIso_95percentBased.txt",
    # neuHadIsolationMapName
    'photonIDValueMapProducer:phoNeutralHadronIsolation' ,
    # neuHadIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/RunIII_Winter22/effectiveArea_NeuHadronIso_95percentBased.txt" ,
    # phoIsolationMapName
    'photonIDValueMapProducer:phoPhotonIsolation' ,
    # phoIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/RunIII_Winter22/effectiveArea_PhotonIso_95percentBased.txt"
)

clusterIsoInputs = ClusterIsolationCutInputs(
    # trkIsolationMapName
    'photonIDValueMapProducer:phoTrkIsolation' ,
    # trkIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/RunIII_Winter22/effectiveArea_TrackerIso_95percentBased.txt",
    # ecalClusIsolationMapName
    'photonIDValueMapProducer:phoEcalPFClIsolation' ,
    # ecalClusIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/RunIII_Winter22/effectiveArea_ECalClusterIso_95percentBased.txt",
    # hcalClusIsolationMapName
    'photonIDValueMapProducer:phoHcalPFClIsolation' ,
    # hcalClusIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/RunIII_Winter22/effectiveArea_HCalClusterIso_95percentBased.txt"
)

hOverEInputs = HoverECutInputs(
    # hOverEEffAreas
    "RecoEgamma/PhotonIdentification/data/RunIII_Winter22/effectiveArea_coneBasedHoverE_95percentBased.txt"
)

#
# Finally, set up VID configuration for all cuts
#
cutBasedPhotonID_RunIIIWinter22_122X_V1_loose  = configureVIDCutBasedPhoID_V6 ( WP_Loose_EB, WP_Loose_EE, isoInputs, clusterIsoInputs, hOverEInputs)
cutBasedPhotonID_RunIIIWinter22_122X_V1_medium = configureVIDCutBasedPhoID_V6 ( WP_Medium_EB, WP_Medium_EE, isoInputs, clusterIsoInputs, hOverEInputs)
cutBasedPhotonID_RunIIIWinter22_122X_V1_tight  = configureVIDCutBasedPhoID_V6 ( WP_Tight_EB, WP_Tight_EE, isoInputs, clusterIsoInputs, hOverEInputs)

## The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to
# 1) comment out the lines below about the registry,
# 2) run "calculateIdMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(cutBasedPhotonID_RunIIIWinter22_122X_V1_loose.idName,
                             '57d3fe8d9a1bff37aca5d13887138607233af1b5')
central_id_registry.register(cutBasedPhotonID_RunIIIWinter22_122X_V1_medium.idName,
                             '114b047ad28e2aae54869847420514d74f2540b8')
central_id_registry.register(cutBasedPhotonID_RunIIIWinter22_122X_V1_tight.idName,
                             '2e56bbbca90e9bc089e5a716412cc51f3de47cb3')

cutBasedPhotonID_RunIIIWinter22_122X_V1_loose.isPOGApproved = cms.untracked.bool(True)
cutBasedPhotonID_RunIIIWinter22_122X_V1_medium.isPOGApproved = cms.untracked.bool(True)
cutBasedPhotonID_RunIIIWinter22_122X_V1_tight.isPOGApproved = cms.untracked.bool(True)

