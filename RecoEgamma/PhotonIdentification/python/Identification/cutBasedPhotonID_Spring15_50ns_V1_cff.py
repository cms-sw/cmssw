
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
#       https://indico.cern.ch/event/369239/contribution/2/attachments/1134693/1623149/spring15_pcb.pdf

#
# First, define cut values
#

# Loose working point Barrel and Endcap
idName = "cutBasedPhotonID-Spring15-50ns-V1-standalone-loose"
WP_Loose_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.05      ,  # hOverECut
    0.0103    ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + exp(pt*C2+C3)
    2.44      ,  # absPFChaHadIsoWithEACut_C1
    0         ,  # absPFChaHadIsoWithEACut_C2
    2.57      ,  # absPFNeuHadIsoWithEACut_C1
    0.0044    ,  # absPFNeuHadIsoWithEACut_C2
    0.5809    ,  # absPFNeuHadIsoWithEACut_C3
    1.92      ,  # absPFPhoIsoWithEACut_C1
    0.0043       # absPFPhoIsoWithEACut_C2
    )
WP_Loose_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.05      ,  # hOverECut
    0.0277    ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + exp(pt*C2+C3)
    1.84      ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    4.00      ,  # absPFNeuHadIsoWithEACut_C1
    0.0040    ,  # absPFNeuHadIsoWithEACut_C2
    0.9402    ,  # absPFNeuHadIsoWithEACut_C3
    2.15      ,  # absPFPhoIsoWithEACut_C1
    0.0041       # absPFPhoIsoWithEACut_C2
    )

# Medium working point Barrel and Endcap
idName = "cutBasedPhotonID-Spring15-50ns-V1-standalone-medium"
WP_Medium_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.05      ,  # hOverECut
    0.0100    ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + exp(pt*C2+C3)
    1.31      ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    0.60      ,  # absPFNeuHadIsoWithEACut_C1
    0.0044    ,  # absPFNeuHadIsoWithEACut_C2
    0.5809    ,  # absPFNeuHadIsowithEACut_C3 
    1.33      ,  # absPFPhoIsoWithEACut_C1
    0.0043       # absPFPhoIsoWithEACut_C2
    )

WP_Medium_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.05      ,  # hOverECut
    0.0267    ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + exp(pt*C2+C3)
    1.25      ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    1.65      ,  # absPFNeuHadIsoWithEACut_C1
    0.0040    ,  # absPFNeuHadIsoWithEACut_C2
    0.9402    ,  # absPFNeuHadIsowithEACut_C3 
    1.02      ,  # absPFPhoIsoWithEACut_C1
    0.0041       # absPFPhoIsoWithEACut_C2
    )

# Tight working point Barrel and Endcap
idName = "cutBasedPhotonID-Spring15-50ns-V1-standalone-tight"
WP_Tight_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.05      ,  # hOverECut
    0.0100    ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + exp(pt*C2+C3)
    0.91      ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    0.33      ,  # absPFNeuHadIsoWithEACut_C1
    0.0044    ,  # absPFNeuHadIsoWithEACut_C2
    0.5809    ,  # absPFNeuHadIsowithEACut_C3
    0.61      ,  # absPFPhoIsoWithEACut_C1
    0.0043        # absPFPhoIsoWithEACut_C2
    )

WP_Tight_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.05      ,  # hOverECut
    0.0267    ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + exp(pt*C2+C3)
    0.65      ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    0.93      ,  # absPFNeuHadIsoWithEACut_C1
    0.0040    ,  # absPFNeuHadIsoWithEACut_C2
    0.9402    ,  # absPFNeuHadIsowithEACut_C3    
    0.54      ,  # absPFPhoIsoWithEACut_C1
    0.0041       # absPFPhoIsoWithEACut_C2
    )


# Second, define where to find the precomputed isolations and what effective
# areas to use for pile-up correction
isoInputs = IsolationCutInputs(
    # chHadIsolationMapName  
    'photonIDValueMapProducer:phoChargedIsolation' ,
    # chHadIsolationEffAreas 
    "RecoEgamma/PhotonIdentification/data/Spring15/effAreaPhotons_cone03_pfChargedHadrons_50ns.txt" ,
    # neuHadIsolationMapName
    'photonIDValueMapProducer:phoNeutralHadronIsolation' ,
    # neuHadIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/Spring15/effAreaPhotons_cone03_pfNeutralHadrons_50ns.txt" ,
    # phoIsolationMapName  
    "photonIDValueMapProducer:phoPhotonIsolation" ,
    # phoIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/Spring15/effAreaPhotons_cone03_pfPhotons_50ns.txt"
)

#
# Finally, set up VID configuration for all cuts
#
cutBasedPhotonID_Spring15_50ns_V1_standalone_loose  = configureVIDCutBasedPhoID_V4 ( WP_Loose_EB, WP_Loose_EE, isoInputs)
cutBasedPhotonID_Spring15_50ns_V1_standalone_medium = configureVIDCutBasedPhoID_V4 ( WP_Medium_EB, WP_Medium_EE, isoInputs)
cutBasedPhotonID_Spring15_50ns_V1_standalone_tight  = configureVIDCutBasedPhoID_V4 ( WP_Tight_EB, WP_Tight_EE, isoInputs)

## The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(cutBasedPhotonID_Spring15_50ns_V1_standalone_loose.idName,
                             '0ecdbf2d9e48f89051f6dd7f9e6547e8')
central_id_registry.register(cutBasedPhotonID_Spring15_50ns_V1_standalone_medium.idName,
                             '9efd6037fab2ff3bb65b7592fac5efde')
central_id_registry.register(cutBasedPhotonID_Spring15_50ns_V1_standalone_tight.idName,
                             'e6a9a77984738862cda3a2fd966fc05e')

cutBasedPhotonID_Spring15_50ns_V1_standalone_loose.isPOGApproved = cms.untracked.bool(True)
cutBasedPhotonID_Spring15_50ns_V1_standalone_medium.isPOGApproved = cms.untracked.bool(True)
cutBasedPhotonID_Spring15_50ns_V1_standalone_tight.isPOGApproved = cms.untracked.bool(True)
