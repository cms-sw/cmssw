import FWCore.ParameterSet.Config as cms

import Geometry.HcalEventSetup.hcalTopologyIdeal_cfi


from RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi import *
from RecoJets.Configuration.CaloTowersRec_cff import *
#from RecoParticleFlow.PFClusterProducer.towerMakerPF_cfi import *
#from RecoParticleFlow.PFClusterProducer.particleFlowCaloResolution_cfi import _timeResolutionHCALMaxSample

from RecoParticleFlow.PFClusterProducer.particleFlowRecHitECAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHE_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHF_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHO_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitPS_cfi import *

from RecoParticleFlow.PFClusterProducer.particleFlowClusterECALUncorrected_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterECAL_cfi import *


from RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHE_cfi import *
#from RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHETimeSelected_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHF_cfi import *

from RecoParticleFlow.PFClusterProducer.particleFlowClusterHCAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHO_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterPS_cfi import *

#multi-depth 
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHEHO_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHCALSemi3D_cfi import *
#hf from rechits
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHF_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHF_cfi import *

from RecoParticleFlow.PFClusterProducer.particleFlowRecHitECALWithTime_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterECALWithTime_cff import *

#gives particleFlowRecHit(Cluster)HGC sequence
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGC_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHGC_cff import *

#provides particleFlowRecHitEK
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitShashlik_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterShashlik_cfi import *

#provides primary vertex
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import *

pfClusteringECAL = cms.Sequence()

pfClusteringECAL = cms.Sequence(particleFlowRecHitECAL*particleFlowClusterECALUncorrected*particleFlowClusterECAL)

pfClusteringEK = cms.Sequence( particleFlowRecHitEK +
                               particleFlowClusterEKUncorrected )

pfClusteringHCAL = cms.Sequence()

#semi3DHCAL=True
#if semi3DHCAL:
#pfClusteringHCAL = cms.Sequence( particleFlowRecHitHBHEHO      +
#                                 particleFlowRecHitHF          +
#                                 particleFlowClusterHCALSemi3D +
#                                 particleFlowClusterHF           )
#else:

towerMakerPF = calotowermaker.clone()
#pfClusteringHCAL = cms.Sequence( towerMakerPF             +
#                                 particleFlowRecHitHCAL   +
#                                 particleFlowRecHitHO     +
#                                 particleFlowClusterHCAL  +
#                                 particleFlowClusterHFHAD +
#                                 particleFlowClusterHFEM  +
#                                 particleFlowClusterHO      )

#pfClusteringHCAL = cms.Sequence(particleFlowRecHitHCAL*particleFlowClusterHCAL)
#pfClusteringHCALall = cms.Sequence(particleFlowClusterHCAL+particleFlowClusterHFHAD+particleFlowClusterHFEM)
#pfClusteringHCAL = cms.Sequence(particleFlowRecHitHCAL*pfClusteringHCALall)

#pfClusteringHO = cms.Sequence(particleFlowRecHitHO*particleFlowClusterHO)

#pfClusteringHCAL = cms.Sequence(particleFlowRecHitHCAL*particleFlowClusterHCAL*particleFlowClusterHFHAD*particleFlowClusterHFEM)
pfClusteringPS = cms.Sequence(particleFlowRecHitPS*particleFlowClusterPS)


#pfClusteringHBHEHF = cms.Sequence(towerMakerPF*particleFlowRecHitHCAL*particleFlowClusterHCAL+particleFlowClusterHFHAD+particleFlowClusterHFEM)
pfClusteringHBHEHF = cms.Sequence(offlinePrimaryVertices*particleFlowRecHitHBHE*particleFlowRecHitHF*particleFlowClusterHBHE*particleFlowClusterHF*particleFlowClusterHCAL)
pfClusteringHO = cms.Sequence(particleFlowRecHitHO*particleFlowClusterHO)


# Changed values
# Don't use calotowers for HO, instead we use RecHits directly, and perform the links with tracks and HCAL clusters.
towerMakerPF.UseHO = False
# Energy threshold for HB Cell inclusion
towerMakerPF.HBThreshold = 0.4
# Energy threshold for HE (S = 5 degree, D = 10 degree, in phi) Cell inclusion
towerMakerPF.HESThreshold = 0.4
towerMakerPF.HEDThreshold = 0.4

# Default values
# Energy threshold for HO cell inclusion [GeV]
towerMakerPF.HOThreshold0 = 0.2
towerMakerPF.HOThresholdPlus1 = 0.8
towerMakerPF.HOThresholdMinus1 = 0.8
towerMakerPF.HOThresholdPlus2 = 0.8
towerMakerPF.HOThresholdMinus2 = 0.8
# Weighting factor for HO 
towerMakerPF.HOWeight = 1.0
towerMakerPF.HOWeights = (1.0, 1.0, 1.0, 1.0, 1.0)
# Weighting factor for HF short-fiber readouts
towerMakerPF.HF2Weight = 1.0
towerMakerPF.HF2Weights = (1.0, 1.0, 1.0, 1.0, 1.0)
# Weighting factor for HF long-fiber readouts 
towerMakerPF.HF1Weight = 1.0
towerMakerPF.HF1Weights = (1.0, 1.0, 1.0, 1.0, 1.0)
# Energy threshold for long-fiber HF readout inclusion [GeV]
towerMakerPF.HF1Threshold = 1.2
# Energy threshold for short-fiber HF readout inclusion [GeV]
towerMakerPF.HF2Threshold = 1.8
# Weighting factor for HB  cells   
towerMakerPF.HBWeight = 1.0
towerMakerPF.HBWeights = (1.0, 1.0, 1.0, 1.0, 1.0)
# Weighting factor for HE 5-degree cells   
towerMakerPF.HESWeight = 1.0
towerMakerPF.HESWeights = (1.0, 1.0, 1.0, 1.0, 1.0)
# Weighting factor for HE 10-degree cells   
towerMakerPF.HEDWeight = 1.0
towerMakerPF.HEDWeights = (1.0, 1.0, 1.0, 1.0, 1.0)
# Global energy threshold on Hcal [GeV]
towerMakerPF.HcalThreshold = -1000.0
# Global energy threshold on tower [GeV]
towerMakerPF.EcutTower = -1000.0
# parameters for handling of anomalous cells
# acceptable severity level
towerMakerPF.HcalAcceptSeverityLevel = 11
# use of recovered hits
towerMakerPF.UseHcalRecoveredHits = True
# flag to allow/disallow missing inputs
towerMakerPF.AllowMissingInputs = False

particleFlowClusterWithoutHO = cms.Sequence(
    #caloTowersRec*
    #towerMakerPF*
    pfClusteringPS*
    pfClusteringECAL*
    pfClusteringHBHEHF
)

particleFlowCluster = cms.Sequence(
    #caloTowersRec*
    #towerMakerPF*
    pfClusteringPS*
    pfClusteringECAL*
    pfClusteringHBHEHF*
    pfClusteringHO 
)


