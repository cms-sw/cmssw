
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Common functions and classes for ID definition are imported here:
from RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_tools import *

#
# This is the first version of Spring15  cuts, optimized on Spring15 50ns samples. 
#
# The ID cuts below are optimized IDs for Spring 50ns Scenario
# The cut values are taken from the twiki:
#       https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedPhotonIdentificationRun2
#       (where they may not stay, if a newer version of cuts becomes available for these
#        conditions)
# See also the presentation explaining these working points (this will not change):
#       ... not yet presented ...

#
# First, define cut values
#

# Loose working point Barrel and Endcap
idName = "cutBasedPhotonID-Spring15-25ns-V1-standalone-loose"
WP_Loose_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.05      ,  # hOverECut
    0.0102    ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    3.32      ,  # absPFChaHadIsoWithEACut_C1
    0         ,  # absPFChaHadIsoWithEACut_C2
    1.92      ,  # absPFNeuHadIsoWithEACut_C1
    0.014     ,  # absPFNeuHadIsoWithEACut_C2
    0.000019  ,  # absPFNeuHadIsoWithEACut_C3
    0.81      ,  # absPFPhoIsoWithEACut_C1
    0.0053       # absPFPhoIsoWithEACut_C2
    )
WP_Loose_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.05      ,  # hOverECut
    0.0274    ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    1.97      ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    11.86     ,  # absPFNeuHadIsoWithEACut_C1
    0.0139    ,  # absPFNeuHadIsoWithEACut_C2
    0.000025  ,  # absPFNeuHadIsoWithEACut_C3
    0.83      ,  # absPFPhoIsoWithEACut_C1
    0.0034       # absPFPhoIsoWithEACut_C2
    )

# Medium working point Barrel and Endcap
idName = "cutBasedPhotonID-Spring15-25ns-V1-standalone-medium"
WP_Medium_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.05      ,  # hOverECut
    0.0102    ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    1.37      ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    1.06      ,  # absPFNeuHadIsoWithEACut_C1
    0.014     ,  # absPFNeuHadIsoWithEACut_C2
    0.000019  ,  # absPFNeuHadIsowithEACut_C3 
    0.28      ,  # absPFPhoIsoWithEACut_C1
    0.0053       # absPFPhoIsoWithEACut_C2
    )

WP_Medium_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.05      ,  # hOverECut
    0.0268    ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    1.10      ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    2.69      ,  # absPFNeuHadIsoWithEACut_C1
    0.0139    ,  # absPFNeuHadIsoWithEACut_C2
    0.000025  ,  # absPFNeuHadIsowithEACut_C3 
    0.39      ,  # absPFPhoIsoWithEACut_C1
    0.0034       # absPFPhoIsoWithEACut_C2
    )

# Tight working point Barrel and Endcap
idName = "cutBasedPhotonID-Spring15-25ns-V1-standalone-tight"
WP_Tight_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.05      ,  # hOverECut
    0.0100    ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    0.76      ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    0.97      ,  # absPFNeuHadIsoWithEACut_C1
    0.014     ,  # absPFNeuHadIsoWithEACut_C2
    0.000019  ,  # absPFNeuHadIsowithEACut_C3
    0.08      ,  # absPFPhoIsoWithEACut_C1
    0.0053       # absPFPhoIsoWithEACut_C2
    )

WP_Tight_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.05      ,  # hOverECut
    0.0268    ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    0.56      ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    2.09      ,  # absPFNeuHadIsoWithEACut_C1
    0.0139    ,  # absPFNeuHadIsoWithEACut_C2
    0.000025  ,  # absPFNeuHadIsowithEACut_C3    
    0.16      ,  # absPFPhoIsoWithEACut_C1
    0.0034       # absPFPhoIsoWithEACut_C2
    )


# Second, define where to find the precomputed isolations and what effective
# areas to use for pile-up correction
isoInputs = IsolationCutInputs(
    # chHadIsolationMapName  
    'photonIDValueMapProducer:phoChargedIsolation' ,
    # chHadIsolationEffAreas 
    "RecoEgamma/PhotonIdentification/data/Spring15/effAreaPhotons_cone03_pfChargedHadrons_25ns_NULLcorrection.txt",
    # neuHadIsolationMapName
    'photonIDValueMapProducer:phoNeutralHadronIsolation' ,
    # neuHadIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/Spring15/effAreaPhotons_cone03_pfNeutralHadrons_25ns_90percentBased.txt" ,
    # phoIsolationMapName  
    "photonIDValueMapProducer:phoPhotonIsolation" ,
    # phoIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/Spring15/effAreaPhotons_cone03_pfPhotons_25ns_90percentBased.txt"
)

#
# Finally, set up VID configuration for all cuts
#
cutBasedPhotonID_Spring15_25ns_V1_standalone_loose  = configureVIDCutBasedPhoID_V5 ( WP_Loose_EB, WP_Loose_EE, isoInputs)
cutBasedPhotonID_Spring15_25ns_V1_standalone_medium = configureVIDCutBasedPhoID_V5 ( WP_Medium_EB, WP_Medium_EE, isoInputs)
cutBasedPhotonID_Spring15_25ns_V1_standalone_tight  = configureVIDCutBasedPhoID_V5 ( WP_Tight_EB, WP_Tight_EE, isoInputs)

## The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(cutBasedPhotonID_Spring15_25ns_V1_standalone_loose.idName,
                             '3dbb7c6922f3e1b9eb9cf1c679ff70bb')
central_id_registry.register(cutBasedPhotonID_Spring15_25ns_V1_standalone_medium.idName,
                             '3c31de4198e6c34a0668e11fae83ac21')
central_id_registry.register(cutBasedPhotonID_Spring15_25ns_V1_standalone_tight.idName,
                             '82ed54bcaf3c8d0ac4f2ae51aa8ff37d')

cutBasedPhotonID_Spring15_25ns_V1_standalone_loose.isPOGApproved = cms.untracked.bool(True)
cutBasedPhotonID_Spring15_25ns_V1_standalone_medium.isPOGApproved = cms.untracked.bool(True)
cutBasedPhotonID_Spring15_25ns_V1_standalone_tight.isPOGApproved = cms.untracked.bool(True)
