import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Reconstruction_Data_cff import ecalLocalRecoSequence, pfClusteringPS, pfClusteringECAL, ecalClusters
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalIsolElectron_cff import *

#ecal rechits
from RecoLocalCalo.Configuration.RecoLocalCalo_cff import ecalLocalRecoSequence
recoECALSeq = cms.Sequence( ecalLocalRecoSequence)

from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *
rerecoPFClusteringSeq = cms.Sequence(pfClusteringPS + pfClusteringECAL)

from  RecoEcal.Configuration.RecoEcal_cff import *
from Calibration.EcalCalibAlgos.electronRecalibSCAssociator_cfi import *
ecalClusteringSeq = cms.Sequence(ecalClusters * electronRecalibSCAssociator)


#sandboxRerecoSeq = cms.Sequence(electronRecoSeq * ecalClusteringSeq)
#sandboxPFRerecoSeq = cms.Sequence(electronRecoSeq * rerecoPFClusteringSeq * ecalClusteringSeq)
rerecoECALSeq = cms.Sequence(recoECALSeq * rerecoPFClusteringSeq * ecalClusteringSeq)

############################################### FINAL SEQUENCES
# sequences used in AlCaRecoStreams_cff.py
#redo the preselection of electrons with selectorProducerSeq for recHit reducers: they use the selected objects as input
seqALCARECOEcalRecalElectron = cms.Sequence( rerecoECALSeq * selectorProducerSeq * ALCARECOEcalCalElectronECALSeq)


