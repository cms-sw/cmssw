
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Common functions and classes for ID definition are imported here:
from RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_tools import *

#
# This is the first version of Spring16 cuts for 80X samples
#
# The cut values are taken from the twiki:
#       https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedPhotonIdentificationRun2
#       (where they may not stay, if a newer version of cuts becomes available for these
#        conditions)
# See also the presentation explaining these working points (this will not change):
#       https://indico.cern.ch/event/491517/contributions/2349134/attachments/1359450/2056689/CutBasedPhotonID_24-10-2016.pdf

#
# First, define cut values
#

# Loose working point Barrel and Endcap
idName = "cutBasedPhotonID-Spring16-V1-loose"
WP_Loose_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.05      ,  # hOverECut
    0.01042   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    1.325     ,  # absPFChaHadIsoWithEACut_C1
    0         ,  # absPFChaHadIsoWithEACut_C2
    4.50      ,  # absPFNeuHadIsoWithEACut_C1
    0.0148    ,  # absPFNeuHadIsoWithEACut_C2
    0.000017  ,  # absPFNeuHadIsoWithEACut_C3
    2.554     ,  # absPFPhoIsoWithEACut_C1
    0.0047       # absPFPhoIsoWithEACut_C2
    )
WP_Loose_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.05      ,  # hOverECut
    0.02683   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    1.293     ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    4.187     ,  # absPFNeuHadIsoWithEACut_C1
    0.0163    ,  # absPFNeuHadIsoWithEACut_C2
    0.000014  ,  # absPFNeuHadIsoWithEACut_C3
    3.86      ,  # absPFPhoIsoWithEACut_C1
    0.0034       # absPFPhoIsoWithEACut_C2
    )

# Medium working point Barrel and Endcap
idName = "cutBasedPhotonID-Spring16-V1-medium"
WP_Medium_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.05      ,  # hOverECut
    0.01012   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    0.789     ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    2.364     ,  # absPFNeuHadIsoWithEACut_C1
    0.0148    ,  # absPFNeuHadIsoWithEACut_C2
    0.000017  ,  # absPFNeuHadIsowithEACut_C3 
    0.425     ,  # absPFPhoIsoWithEACut_C1
    0.0047       # absPFPhoIsoWithEACut_C2
    )

WP_Medium_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.05      ,  # hOverECut
    0.02678   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    0.447     ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    1.765     ,  # absPFNeuHadIsoWithEACut_C1
    0.0163    ,  # absPFNeuHadIsoWithEACut_C2
    0.000014  ,  # absPFNeuHadIsowithEACut_C3 
    3.15      ,  # absPFPhoIsoWithEACut_C1
    0.0034       # absPFPhoIsoWithEACut_C2
    )

# Tight working point Barrel and Endcap
idName = "cutBasedPhotonID-Spring16-V1-tight"
WP_Tight_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.05      ,  # hOverECut
    0.01012   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    0.227     ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    1.691     ,  # absPFNeuHadIsoWithEACut_C1
    0.0148    ,  # absPFNeuHadIsoWithEACut_C2
    0.000017  ,  # absPFNeuHadIsowithEACut_C3
    0.346     ,  # absPFPhoIsoWithEACut_C1
    0.0047       # absPFPhoIsoWithEACut_C2
    )

WP_Tight_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.05      ,  # hOverECut
    0.02649   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    0.146     ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    0.432     ,  # absPFNeuHadIsoWithEACut_C1
    0.0163    ,  # absPFNeuHadIsoWithEACut_C2
    0.000014  ,  # absPFNeuHadIsowithEACut_C3    
    2.75      ,  # absPFPhoIsoWithEACut_C1
    0.0034       # absPFPhoIsoWithEACut_C2
    )


# Second, define where to find the precomputed isolations and what effective
# areas to use for pile-up correction
isoInputs = IsolationCutInputs(
    # chHadIsolationMapName  
    'photonIDValueMapProducer:phoChargedIsolation' ,
    # chHadIsolationEffAreas 
    "RecoEgamma/PhotonIdentification/data/Spring16/effAreaPhotons_cone03_pfChargedHadrons_90percentBased.txt",
    # neuHadIsolationMapName
    'photonIDValueMapProducer:phoNeutralHadronIsolation' ,
    # neuHadIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/Spring16/effAreaPhotons_cone03_pfNeutralHadrons_90percentBased.txt" ,
    # phoIsolationMapName  
    "photonIDValueMapProducer:phoPhotonIsolation" ,
    # phoIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/Spring16/effAreaPhotons_cone03_pfPhotons_90percentBased.txt"
)

#
# Finally, set up VID configuration for all cuts
#
cutBasedPhotonID_Spring16_V1_loose  = configureVIDCutBasedPhoID_V5 ( WP_Loose_EB, WP_Loose_EE, isoInputs)
cutBasedPhotonID_Spring16_V1_medium = configureVIDCutBasedPhoID_V5 ( WP_Medium_EB, WP_Medium_EE, isoInputs)
cutBasedPhotonID_Spring16_V1_tight  = configureVIDCutBasedPhoID_V5 ( WP_Tight_EB, WP_Tight_EE, isoInputs)

## The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(cutBasedPhotonID_Spring16_V1_loose.idName,
                             '651f4061bc8c6065d57dba242da2c7c5')
central_id_registry.register(cutBasedPhotonID_Spring16_V1_medium.idName,
                             'f7b30d03b54bfcd6d01ab5e32f3ae910')
central_id_registry.register(cutBasedPhotonID_Spring16_V1_tight.idName,
                             '597d1e862cf598aa3fb208d75ff2ff86')

cutBasedPhotonID_Spring16_V1_loose.isPOGApproved = cms.untracked.bool(True)
cutBasedPhotonID_Spring16_V1_medium.isPOGApproved = cms.untracked.bool(True)
cutBasedPhotonID_Spring16_V1_tight.isPOGApproved = cms.untracked.bool(True)
