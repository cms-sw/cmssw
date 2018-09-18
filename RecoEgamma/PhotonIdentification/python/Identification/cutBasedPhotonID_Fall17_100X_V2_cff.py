from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

import FWCore.ParameterSet.Config as cms

# Common functions and classes for ID definition are imported here:
from RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_tools \
    import ( WorkingPoint_V2,
             IsolationCutInputs,
             configureVIDCutBasedPhoID_V5 )             

# The cut values are taken from the twiki:
#       https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedPhotonIdentificationRun2
#       (where they may not stay, if a newer version of cuts becomes available for these
#        conditions)

# See also the presentation explaining these working points (this will not change):
#     https://indico.cern.ch/event/732974/contributions/3072291/attachments/1685029/2709189/PhotonIDStudy.pdf


#
# First, define cut values
#

# Loose working point Barrel and Endcap
idName = "cutBasedPhotonID-Fall17-100X-V2-loose"
WP_Loose_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.04596    ,  # hOverECut
    0.0106   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    1.694     ,  # absPFChaHadIsoWithEACut_C1
    0         ,  # absPFChaHadIsoWithEACut_C2
    24.032    ,  # absPFNeuHadIsoWithEACut_C1
    0.01512   ,  # absPFNeuHadIsoWithEACut_C2
    0.00002259 ,  # absPFNeuHadIsoWithEACut_C3
    2.876      ,  # absPFPhoIsoWithEACut_C1
    0.004017      # absPFPhoIsoWithEACut_C2
    )
WP_Loose_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.0590    ,  # hOverECut
    0.0272   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    2.089    ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    19.722   ,  # absPFNeuHadIsoWithEACut_C1
    0.0117    ,  # absPFNeuHadIsoWithEACut_C2
    0.000023  ,  # absPFNeuHadIsoWithEACut_C3
    4.162     ,  # absPFPhoIsoWithEACut_C1
    0.0037       # absPFPhoIsoWithEACut_C2
    )

# Medium working point Barrel and Endcap
idName = "cutBasedPhotonID-Fall17-100X-V2-medium"
WP_Medium_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.02197      ,  # hOverECut
    0.01015      ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    1.141    ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    1.189    ,  # absPFNeuHadIsoWithEACut_C1
    0.01512   ,  # absPFNeuHadIsoWithEACut_C2
    0.00002259 ,  # absPFNeuHadIsowithEACut_C3 
    2.08     ,  # absPFPhoIsoWithEACut_C1
    0.004017      # absPFPhoIsoWithEACut_C2
    )

WP_Medium_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.0326    ,  # hOverECut
    0.0272    ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    1.051    ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    2.718    ,  # absPFNeuHadIsoWithEACut_C1
    0.0117    ,  # absPFNeuHadIsoWithEACut_C2
    0.000023  ,  # absPFNeuHadIsowithEACut_C3 
    3.867    ,  # absPFPhoIsoWithEACut_C1
    0.0037       # absPFPhoIsoWithEACut_C2
    )

# Tight working point Barrel and Endcap
idName = "cutBasedPhotonID-Fall17-100X-V2-tight"
WP_Tight_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.02148      ,  # hOverECut
    0.00996      ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    0.65    ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    0.317     ,  # absPFNeuHadIsoWithEACut_C1
    0.01512    ,  # absPFNeuHadIsoWithEACut_C2
    0.00002259 ,  # absPFNeuHadIsowithEACut_C3
    2.044     ,  # absPFPhoIsoWithEACut_C1
    0.004017      # absPFPhoIsoWithEACut_C2
    )

WP_Tight_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.0321    ,  # hOverECut
    0.0271   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    0.517    ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    2.716    ,  # absPFNeuHadIsoWithEACut_C1
    0.0117    ,  # absPFNeuHadIsoWithEACut_C2
    0.000023  ,  # absPFNeuHadIsowithEACut_C3    
    3.032     ,  # absPFPhoIsoWithEACut_C1
    0.0037       # absPFPhoIsoWithEACut_C2
    )


# Second, define where to find the precomputed isolations and what effective
# areas to use for pile-up correction
isoInputs = IsolationCutInputs(
    # chHadIsolationMapName  
    'photonIDValueMapProducer:phoChargedIsolation' ,
    # chHadIsolationEffAreas 
    "RecoEgamma/PhotonIdentification/data/Fall17/effAreaPhotons_cone03_pfChargedHadrons_90percentBased_V2.txt",
    # neuHadIsolationMapName
    'photonIDValueMapProducer:phoNeutralHadronIsolation' ,
    # neuHadIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/Fall17/effAreaPhotons_cone03_pfNeutralHadrons_90percentBased_V2.txt" ,
    # phoIsolationMapName  
    "photonIDValueMapProducer:phoPhotonIsolation" ,
    # phoIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/Fall17/effAreaPhotons_cone03_pfPhotons_90percentBased_V2.txt"
)

#
# Finally, set up VID configuration for all cuts
#
cutBasedPhotonID_Fall17_100X_V2_loose  = configureVIDCutBasedPhoID_V5 ( WP_Loose_EB, WP_Loose_EE, isoInputs)
cutBasedPhotonID_Fall17_100X_V2_medium = configureVIDCutBasedPhoID_V5 ( WP_Medium_EB, WP_Medium_EE, isoInputs)
cutBasedPhotonID_Fall17_100X_V2_tight  = configureVIDCutBasedPhoID_V5 ( WP_Tight_EB, WP_Tight_EE, isoInputs)

## The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(cutBasedPhotonID_Fall17_100X_V2_loose.idName,
                             '4578dfcceb0bfd1ba5ac28973c843fd0')
central_id_registry.register(cutBasedPhotonID_Fall17_100X_V2_medium.idName,
                             '28b186c301061395f394a81266c8d7de')
central_id_registry.register(cutBasedPhotonID_Fall17_100X_V2_tight.idName,
                             '6f4f0ed6a8bf2de8dcf0bc3349b0546d')

cutBasedPhotonID_Fall17_100X_V2_loose.isPOGApproved = cms.untracked.bool(True)
cutBasedPhotonID_Fall17_100X_V2_medium.isPOGApproved = cms.untracked.bool(True)
cutBasedPhotonID_Fall17_100X_V2_tight.isPOGApproved = cms.untracked.bool(True)
