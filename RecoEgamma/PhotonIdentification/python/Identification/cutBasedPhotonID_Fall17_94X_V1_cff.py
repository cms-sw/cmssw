
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

import FWCore.ParameterSet.Config as cms

# Common functions and classes for ID definition are imported here:
from RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_tools \
    import ( WorkingPoint_V2,
             IsolationCutInputs,
             configureVIDCutBasedPhoID_V5 )             

#
# This is the first version of Spring16 cuts for 80X samples
#
# The cut values are taken from the twiki:
#       https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedPhotonIdentificationRun2
#       (where they may not stay, if a newer version of cuts becomes available for these
#        conditions)
# See also the presentation explaining these working points (this will not change):
# https://indico.cern.ch/event/662751/contributions/2778043/attachments/1562017/2459674/EGamma_WorkShop_21.11.17_Debabrata.pdf 

#
# First, define cut values
#

# Loose working point Barrel and Endcap
idName = "cutBasedPhotonID-Fall17-94X-V1-loose"
WP_Loose_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.043    ,  # hOverECut
    0.0101   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    1.403     ,  # absPFChaHadIsoWithEACut_C1
    0         ,  # absPFChaHadIsoWithEACut_C2
    15.959    ,  # absPFNeuHadIsoWithEACut_C1
    0.0127    ,  # absPFNeuHadIsoWithEACut_C2
    0.000026  ,  # absPFNeuHadIsoWithEACut_C3
    3.06     ,  # absPFPhoIsoWithEACut_C1
    0.0038       # absPFPhoIsoWithEACut_C2
    )
WP_Loose_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.026    ,  # hOverECut
    0.0267   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    2.809     ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    7.056     ,  # absPFNeuHadIsoWithEACut_C1
    0.0117    ,  # absPFNeuHadIsoWithEACut_C2
    0.000025  ,  # absPFNeuHadIsoWithEACut_C3
    4.766     ,  # absPFPhoIsoWithEACut_C1
    0.0038       # absPFPhoIsoWithEACut_C2
    )

# Medium working point Barrel and Endcap
idName = "cutBasedPhotonID-Fall17-94X-V1-medium"
WP_Medium_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.032    ,  # hOverECut
    0.0101   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    0.43     ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    2.133     ,  # absPFNeuHadIsoWithEACut_C1
    0.0127    ,  # absPFNeuHadIsoWithEACut_C2
    0.000026  ,  # absPFNeuHadIsowithEACut_C3 
    2.344     ,  # absPFPhoIsoWithEACut_C1
    0.0038       # absPFPhoIsoWithEACut_C2
    )

WP_Medium_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.0219    ,  # hOverECut
    0.03001   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    0.442     ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    1.715     ,  # absPFNeuHadIsoWithEACut_C1
    0.0117    ,  # absPFNeuHadIsoWithEACut_C2
    0.000025  ,  # absPFNeuHadIsowithEACut_C3 
    3.863     ,  # absPFPhoIsoWithEACut_C1
    0.0038       # absPFPhoIsoWithEACut_C2
    )

# Tight working point Barrel and Endcap
idName = "cutBasedPhotonID-Fall17-94X-V1-tight"
WP_Tight_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.022    ,  # hOverECut
    0.0099   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    0.101     ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    0.137     ,  # absPFNeuHadIsoWithEACut_C1
    0.0127    ,  # absPFNeuHadIsoWithEACut_C2
    0.000026  ,  # absPFNeuHadIsowithEACut_C3
    2.308     ,  # absPFPhoIsoWithEACut_C1
    0.0038       # absPFPhoIsoWithEACut_C2
    )

WP_Tight_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.021    ,  # hOverECut
    0.0267   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    0.134     ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    1.615     ,  # absPFNeuHadIsoWithEACut_C1
    0.0117    ,  # absPFNeuHadIsoWithEACut_C2
    0.000025  ,  # absPFNeuHadIsowithEACut_C3    
    3.107     ,  # absPFPhoIsoWithEACut_C1
    0.0038       # absPFPhoIsoWithEACut_C2
    )


# Second, define where to find the precomputed isolations and what effective
# areas to use for pile-up correction
isoInputs = IsolationCutInputs(
    # chHadIsolationMapName  
    'photonIDValueMapProducer:phoChargedIsolation' ,
    # chHadIsolationEffAreas 
    "RecoEgamma/PhotonIdentification/data/Fall17/effAreaPhotons_cone03_pfChargedHadrons_90percentBased.txt",
    # neuHadIsolationMapName
    'photonIDValueMapProducer:phoNeutralHadronIsolation' ,
    # neuHadIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/Fall17/effAreaPhotons_cone03_pfNeutralHadrons_90percentBased.txt" ,
    # phoIsolationMapName  
    "photonIDValueMapProducer:phoPhotonIsolation" ,
    # phoIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/Fall17/effAreaPhotons_cone03_pfPhotons_90percentBased.txt"
)

#
# Finally, set up VID configuration for all cuts
#
cutBasedPhotonID_Fall17_94X_V1_loose  = configureVIDCutBasedPhoID_V5 ( WP_Loose_EB, WP_Loose_EE, isoInputs)
cutBasedPhotonID_Fall17_94X_V1_medium = configureVIDCutBasedPhoID_V5 ( WP_Medium_EB, WP_Medium_EE, isoInputs)
cutBasedPhotonID_Fall17_94X_V1_tight  = configureVIDCutBasedPhoID_V5 ( WP_Tight_EB, WP_Tight_EE, isoInputs)

## The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(cutBasedPhotonID_Fall17_94X_V1_loose.idName,
                             '08547098f52eb608b545953f02583c3f')
central_id_registry.register(cutBasedPhotonID_Fall17_94X_V1_medium.idName,
                             'fb58ccd713d6be1f86f1d2e48c69e401')
central_id_registry.register(cutBasedPhotonID_Fall17_94X_V1_tight.idName,
                             '296da1cdbf6f35a99287c5a527472ed3')
cutBasedPhotonID_Fall17_94X_V1_loose.isPOGApproved = cms.untracked.bool(True)
cutBasedPhotonID_Fall17_94X_V1_medium.isPOGApproved = cms.untracked.bool(True)
cutBasedPhotonID_Fall17_94X_V1_tight.isPOGApproved = cms.untracked.bool(True)
