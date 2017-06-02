
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
#     https://indico.cern.ch/event/491548/contributions/2384977/attachments/1377936/2102914/CutBasedPhotonID_25-11-2016.pdf

#
# First, define cut values
#

# Loose working point Barrel and Endcap
idName = "cutBasedPhotonID-Spring16-V2p2-loose"
WP_Loose_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.0597    ,  # hOverECut
    0.01031   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    1.295     ,  # absPFChaHadIsoWithEACut_C1
    0         ,  # absPFChaHadIsoWithEACut_C2
    10.910    ,  # absPFNeuHadIsoWithEACut_C1
    0.0148    ,  # absPFNeuHadIsoWithEACut_C2
    0.000017  ,  # absPFNeuHadIsoWithEACut_C3
    3.630     ,  # absPFPhoIsoWithEACut_C1
    0.0047       # absPFPhoIsoWithEACut_C2
    )
WP_Loose_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.0481    ,  # hOverECut
    0.03013   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    1.011     ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    5.931     ,  # absPFNeuHadIsoWithEACut_C1
    0.0163    ,  # absPFNeuHadIsoWithEACut_C2
    0.000014  ,  # absPFNeuHadIsoWithEACut_C3
    6.641     ,  # absPFPhoIsoWithEACut_C1
    0.0034       # absPFPhoIsoWithEACut_C2
    )

# Medium working point Barrel and Endcap
idName = "cutBasedPhotonID-Spring16-V2p2-medium"
WP_Medium_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.0396    ,  # hOverECut
    0.01022   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    0.441     ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    2.725     ,  # absPFNeuHadIsoWithEACut_C1
    0.0148    ,  # absPFNeuHadIsoWithEACut_C2
    0.000017  ,  # absPFNeuHadIsowithEACut_C3 
    2.571     ,  # absPFPhoIsoWithEACut_C1
    0.0047       # absPFPhoIsoWithEACut_C2
    )

WP_Medium_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.0219    ,  # hOverECut
    0.03001   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    0.442     ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    1.715     ,  # absPFNeuHadIsoWithEACut_C1
    0.0163    ,  # absPFNeuHadIsoWithEACut_C2
    0.000014  ,  # absPFNeuHadIsowithEACut_C3 
    3.863     ,  # absPFPhoIsoWithEACut_C1
    0.0034       # absPFPhoIsoWithEACut_C2
    )

# Tight working point Barrel and Endcap
idName = "cutBasedPhotonID-Spring16-V2p2-tight"
WP_Tight_EB = WorkingPoint_V2(
    idName    ,  # idName
    0.0269    ,  # hOverECut
    0.00994   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    0.202     ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    0.264     ,  # absPFNeuHadIsoWithEACut_C1
    0.0148    ,  # absPFNeuHadIsoWithEACut_C2
    0.000017  ,  # absPFNeuHadIsowithEACut_C3
    2.362     ,  # absPFPhoIsoWithEACut_C1
    0.0047       # absPFPhoIsoWithEACut_C2
    )

WP_Tight_EE = WorkingPoint_V2(
    idName    ,  #idName
    0.0213    ,  # hOverECut
    0.03000   ,  # full5x5_SigmaIEtaIEtaCut
# Isolation cuts are generally absIso < C1 + pt*C2, except for NeuHad is < C1 + pt*C2 + pt*pt*C3
    0.034     ,  # absPFChaHadIsoWithEACut_C1
    0.00      ,  # absPFChaHadIsoWithEACut_C2
    0.586     ,  # absPFNeuHadIsoWithEACut_C1
    0.0163    ,  # absPFNeuHadIsoWithEACut_C2
    0.000014  ,  # absPFNeuHadIsowithEACut_C3    
    2.617     ,  # absPFPhoIsoWithEACut_C1
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
cutBasedPhotonID_Spring16_V2p2_loose  = configureVIDCutBasedPhoID_V5 ( WP_Loose_EB, WP_Loose_EE, isoInputs)
cutBasedPhotonID_Spring16_V2p2_medium = configureVIDCutBasedPhoID_V5 ( WP_Medium_EB, WP_Medium_EE, isoInputs)
cutBasedPhotonID_Spring16_V2p2_tight  = configureVIDCutBasedPhoID_V5 ( WP_Tight_EB, WP_Tight_EE, isoInputs)

## The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(cutBasedPhotonID_Spring16_V2p2_loose.idName,
                             'd6ce6a4f3476294bf0a3261e00170daf')
central_id_registry.register(cutBasedPhotonID_Spring16_V2p2_medium.idName,
                             'c739cfd0b6287b8586da187c06d4053f')
central_id_registry.register(cutBasedPhotonID_Spring16_V2p2_tight.idName,
                             'bdb623bdb1a15c13545020a919dd9530')

cutBasedPhotonID_Spring16_V2p2_loose.isPOGApproved = cms.untracked.bool(True)
cutBasedPhotonID_Spring16_V2p2_medium.isPOGApproved = cms.untracked.bool(True)
cutBasedPhotonID_Spring16_V2p2_tight.isPOGApproved = cms.untracked.bool(True)
