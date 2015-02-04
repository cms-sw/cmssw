import FWCore.ParameterSet.Config as cms

#temporarily load regression weights with esprefer
from CondCore.DBCommon.CondDBCommon_cfi import *

pfcluscorsource = cms.ESSource("PoolDBESSource",
    CondDBCommon,
    DumpStat=cms.untracked.bool(False),
    toGet = cms.VPSet(
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EB_pfSize1_mean_50ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EB_pfSize1_mean_50ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EB_pfSize2_mean_50ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EB_pfSize2_mean_50ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EB_pfSize3_mean_50ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EB_pfSize3_mean_50ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EE_pfSize1_mean_50ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EE_pfSize1_mean_50ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EE_pfSize2_mean_50ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EE_pfSize2_mean_50ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EE_pfSize3_mean_50ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EE_pfSize3_mean_50ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EB_pfSize1_sigma_50ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EB_pfSize1_sigma_50ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EB_pfSize2_sigma_50ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EB_pfSize2_sigma_50ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EB_pfSize3_sigma_50ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EB_pfSize3_sigma_50ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EE_pfSize1_sigma_50ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EE_pfSize1_sigma_50ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EE_pfSize2_sigma_50ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EE_pfSize2_sigma_50ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EE_pfSize3_sigma_50ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EE_pfSize3_sigma_50ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EB_pfSize1_mean_25ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EB_pfSize1_mean_25ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EB_pfSize2_mean_25ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EB_pfSize2_mean_25ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EB_pfSize3_mean_25ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EB_pfSize3_mean_25ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EE_pfSize1_mean_25ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EE_pfSize1_mean_25ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EE_pfSize2_mean_25ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EE_pfSize2_mean_25ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EE_pfSize3_mean_25ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EE_pfSize3_mean_25ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EB_pfSize1_sigma_25ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EB_pfSize1_sigma_25ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EB_pfSize2_sigma_25ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EB_pfSize2_sigma_25ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EB_pfSize3_sigma_25ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EB_pfSize3_sigma_25ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EE_pfSize1_sigma_25ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EE_pfSize1_sigma_25ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EE_pfSize2_sigma_25ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EE_pfSize2_sigma_25ns')
      ),
      cms.PSet(
        record = cms.string('GBRDWrapperRcd'),
        label = cms.untracked.string('GBRForestD_ecalPFClusterCor_EE_pfSize3_sigma_25ns'),
        tag = cms.string('GBRForestD_ecalPFClusterCor_EE_pfSize3_sigma_25ns')
      ),
    )
)

pfcluscorsource.connect = cms.string('frontier://FrontierProd/CMS_COND_PAT_000')

pfclusprefer = cms.ESPrefer(
    'PoolDBESSource',
    'pfcluscorsource',
    GBRDWrapperRcd = cms.vstring(
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EB_pfSize1_mean_50ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EB_pfSize2_mean_50ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EB_pfSize3_mean_50ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EE_pfSize1_mean_50ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EE_pfSize2_mean_50ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EE_pfSize3_mean_50ns',     
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EB_pfSize1_sigma_50ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EB_pfSize2_sigma_50ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EB_pfSize3_sigma_50ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EE_pfSize1_sigma_50ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EE_pfSize2_sigma_50ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EE_pfSize3_sigma_50ns',     
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EB_pfSize1_mean_25ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EB_pfSize2_mean_25ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EB_pfSize3_mean_25ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EE_pfSize1_mean_25ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EE_pfSize2_mean_25ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EE_pfSize3_mean_25ns',     
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EB_pfSize1_sigma_25ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EB_pfSize2_sigma_25ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EB_pfSize3_sigma_25ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EE_pfSize1_sigma_25ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EE_pfSize2_sigma_25ns',
                                'GBRForestD/GBRForestD_ecalPFClusterCor_EE_pfSize3_sigma_25ns',                                  
                                )
)


#### PF CLUSTER ECAL ####

#energy corrector for corrected cluster producer
_emEnergyCorrector = cms.PSet(
    algoName = cms.string("PFClusterEMEnergyCorrector"),
    recHitsEBLabel = cms.InputTag('ecalRecHit', 'EcalRecHitsEB'),
    recHitsEELabel = cms.InputTag('ecalRecHit', 'EcalRecHitsEE'),
    verticesLabel = cms.InputTag('offlinePrimaryVertices'),    
    autoDetectBunchSpacing = cms.bool(True),
)

particleFlowClusterECAL = cms.EDProducer(
    "CorrectedECALPFClusterProducer",
    inputECAL = cms.InputTag("particleFlowClusterECALUncorrected"),
    inputPS = cms.InputTag("particleFlowClusterPS"),
    minimumPSEnergy = cms.double(0.0),
    energyCorrector = _emEnergyCorrector
    )

