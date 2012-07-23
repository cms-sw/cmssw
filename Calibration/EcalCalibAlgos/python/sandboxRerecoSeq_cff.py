import FWCore.ParameterSet.Config as cms

        
from RecoLocalCalo.Configuration.RecoLocalCalo_cff import *
#ecalRecHit.EBuncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB","ALCASKIM")
#ecalRecHit.EEuncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE","ALCASKIM")

electronRecoSeq = cms.Sequence( ecalRecHit * ecalCompactTrigPrim * ecalTPSkim * ecalPreshowerRecHit)
    
from  RecoEcal.Configuration.RecoEcal_cff import *
#correctedHybridSuperClusters.corectedSuperClusterCollection = 'recalibSC'
#correctedMulti5x5SuperClustersWithPreshower.corectedSuperClusterCollection = 'endcapRecalibSC'
#    if(re.match("CMSSW_5_.*",CMSSW_VERSION)):
#        multi5x5PreshowerClusterShape.endcapSClusterProducer = "correctedMulti5x5SuperClustersWithPreshower:endcapRecalibSC"

#    process.load("Calibration.EcalCalibAlgos.electronRecalibSCAssociator_cfi")
from Calibration.EcalCalibAlgos.electronRecalibSCAssociator_cfi import *
#electronRecalibSCAssociator.scIslandCollection = cms.string('endcapRecalibSC')
#electronRecalibSCAssociator.scIslandProducer = cms.string('correctedMulti5x5SuperClustersWithPreshower')
#electronRecalibSCAssociator.scProducer = cms.string('correctedHybridSuperClusters')
#electronRecalibSCAssociator.scCollection = cms.string('recalibSC')
#electronRecalibSCAssociator.electronProducer = 'gsfElectrons'
electronClusteringSeq = cms.Sequence(ecalClusters * electronRecalibSCAssociator)


sandboxRerecoSeq = cms.Sequence(electronRecoSeq * electronClusteringSeq)
