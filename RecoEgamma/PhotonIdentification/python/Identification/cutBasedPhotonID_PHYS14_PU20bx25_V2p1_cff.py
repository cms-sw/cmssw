
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Common functions and classes for ID definition are imported here:
from RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_tools import *

## This is the same as V2 except for using the V3 photon ID
#
# This is the first version of PHYS14 cuts, optimized on  PHYS14 samples. 
#
# The ID cuts below are optimized IDs for PHYS14 Scenario PU 20, bx 25ns
# The cut values are taken from the twiki:
#       https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedPhotonIdentificationRun2#PHYS14_selections_PU20_bunch_cro
#       (where they may not stay, if a newer version of cuts becomes available for these
#        conditions)
# See also the presentation explaining these working points (this will not change):
#    https://indico.cern.ch/event/369225/contribution/1/material/slides/0.pdf

#
# First, define cut values
#

# Loose working point Barrel and Endcap
idName = "cutBasedPhotonID-PHYS14-PU20bx25-V2p1-standalone-loose"
WP_Loose_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.028     ,  # hOverECut
    0.0107    ,  # full5x5_SigmaIEtaIEtaCut
    # Isolation cuts are generally absIso < C1 + pt*C2, except for barrel NeuHad is < C1 + exp(pt*C2+C3)
    2.67      ,  # absPFChaHadIsoWithEACut_C1
    0         ,  # absPFChaHadIsoWithEACut_C2
    7.23      ,  # absPFNeuHadIsoWithEACut_C1
    0.0028    ,  # absPFNeuHadIsoWithEACut_C2
    0.5408    ,  # absPFNeuHadIsoWithEACut_C3
    2.11      ,  # absPFPhoIsoWithEACut_C1
    0.0014        # absPFPhoIsoWithEACut_C2
    )

WP_Loose_EE = WorkingPoint_V1(
    idName    ,  #idName
    0.093     ,  # hOverECut
    0.0272   ,  # full5x5_SigmaIEtaIEtaCut
    # Isolation cuts are generally absIso < C1 + pt*C2
    1.79      ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    8.89      ,  # absPFNeuHadIsoWithEACut_C1
    0.01725   ,  # absPFNeuHadIsoWithEACut_C2
    3.09      ,  # absPFPhoIsoWithEACut_C1
    0.0091       # absPFPhoIsoWithEACut_C2
    )

# Medium working point Barrel and Endcap
idName = "cutBasedPhotonID-PHYS14-PU20bx25-V2p1-standalone-medium"
WP_Medium_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.012     ,  # hOverECut
    0.0100    ,  # full5x5_SigmaIEtaIEtaCut
    # Isolation cuts are generally absIso < C1 + pt*C2, except for barrel NeuHad is < C1 + exp(pt*C2+C3)
    1.79      ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    0.16      ,  # absPFNeuHadIsoWithEACut_C1
    0.0028    ,  # absPFNeuHadIsoWithEACut_C2
    0.5408    ,  # absPFNeuHadIsoWithEACut_C3
    1.90      ,  # absPFPhoIsoWithEACut_C1
    0.0014        # absPFPhoIsoWithEACut_C2
    )

WP_Medium_EE = WorkingPoint_V1(
    idName    ,  #idName
    0.023    ,  # hOverECut
    0.0267    ,  # full5x5_SigmaIEtaIEtaCut
    # Isolation cuts are generally absIso < C1 + pt*C2
    1.09     ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    4.31      ,  # absPFNeuHadIsoWithEACut_C1
    0.0172    ,  # absPFNeuHadIsoWithEACut_C2
    1.90      ,  # absPFPhoIsoWithEACut_C1
    0.0091       # absPFPhoIsoWithEACut_C2
    )

# Tight working point Barrel and Endcap
idName = "cutBasedPhotonID-PHYS14-PU20bx25-V2p1-standalone-tight"
WP_Tight_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.010     ,  # hOverECut
    0.0100    ,  # full5x5_SigmaIEtaIEtaCut
    # Isolation cuts are generally absIso < C1 + pt*C2, except for barrel NeuHad is < C1 + exp(pt*C2+C3)
    1.66      ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    0.14      ,  # absPFNeuHadIsoWithEACut_C1
    0.0028    ,  # absPFNeuHadIsoWithEACut_C2
    0.5408    ,  # absPFNeuHadIsoWithEACut_C3
    1.40      ,  # absPFPhoIsoWithEACut_C1
    0.0014        # absPFPhoIsoWithEACut_C2
    )

WP_Tight_EE = WorkingPoint_V1(
    idName    ,  #idName
    0.015     ,  # hOverECut
    0.0265    ,  # full5x5_SigmaIEtaIEtaCut
    # Isolation cuts are generally absIso < C1 + pt*C2
    1.04      ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    3.89      ,  # absPFNeuHadIsoWithEACut_C1
    0.0172    ,  # absPFNeuHadIsoWithEACut_C2
    1.40      ,  # absPFPhoIsoWithEACut_C1
    0.0091       # absPFPhoIsoWithEACut_C2
    )


# Second, define where to find the precomputed isolations and what effective
# areas to use for pile-up correction
isoInputs = IsolationCutInputs(
    # chHadIsolationMapName  
    'photonIDValueMapProducer:phoChargedIsolation' ,
    # chHadIsolationEffAreas 
    "RecoEgamma/PhotonIdentification/data/PHYS14/effAreaPhotons_cone03_pfChargedHadrons_V2.txt" ,
    # neuHadIsolationMapName
    'photonIDValueMapProducer:phoNeutralHadronIsolation' ,
    # neuHadIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/PHYS14/effAreaPhotons_cone03_pfNeutralHadrons_V2.txt" ,
    # phoIsolationMapName  
    "photonIDValueMapProducer:phoPhotonIsolation" ,
    # phoIsolationEffAreas
    "RecoEgamma/PhotonIdentification/data/PHYS14/effAreaPhotons_cone03_pfPhotons_V2.txt"
)

#
# Finally, set up VID configuration for all cuts
#
cutBasedPhotonID_PHYS14_PU20bx25_V2p1_standalone_loose  = configureVIDCutBasedPhoID_V3 ( WP_Loose_EB, WP_Loose_EE, isoInputs)
cutBasedPhotonID_PHYS14_PU20bx25_V2p1_standalone_medium = configureVIDCutBasedPhoID_V3 ( WP_Medium_EB, WP_Medium_EE, isoInputs)
cutBasedPhotonID_PHYS14_PU20bx25_V2p1_standalone_tight  = configureVIDCutBasedPhoID_V3 ( WP_Tight_EB, WP_Tight_EE, isoInputs)

#
# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(cutBasedPhotonID_PHYS14_PU20bx25_V2p1_standalone_loose.idName,
                             '88b068028aebbd139249cd3d82ed4693')
central_id_registry.register(cutBasedPhotonID_PHYS14_PU20bx25_V2p1_standalone_medium.idName,
                             '4c90f02adee9a99a6bb026c0c20c0894')
central_id_registry.register(cutBasedPhotonID_PHYS14_PU20bx25_V2p1_standalone_tight.idName,
                             '3a7ef3805194a788621aeecb70e66b21')

#for now until we have a database...
cutBasedPhotonID_PHYS14_PU20bx25_V2p1_standalone_loose.isPOGApproved = cms.untracked.bool(True)
cutBasedPhotonID_PHYS14_PU20bx25_V2p1_standalone_medium.isPOGApproved = cms.untracked.bool(True)
cutBasedPhotonID_PHYS14_PU20bx25_V2p1_standalone_tight.isPOGApproved = cms.untracked.bool(True)
