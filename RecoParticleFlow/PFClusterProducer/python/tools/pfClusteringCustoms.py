import FWCore.ParameterSet.Config as cms





def customizePFECALClustering(process,scenario):

    if scenario ==1:
        print 'SCENARIO 1: Default ECAL clustering & ECAL time reconstruction'
        print 'Timing cuts are applied to the PF cluster level'
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterECALTimeSelected_cfi')
        process.particleFlowClusterECALTimeSelected.src = cms.InputTag('particleFlowClusterECALUncorrected')
        process.particleFlowClusterECAL.inputECAL = cms.InputTag('particleFlowClusterECALTimeSelected')
        
        i = process.pfClusteringECAL.index(process.particleFlowClusterECALUncorrected)
        process.pfClusteringECAL.insert(i+1,process.particleFlowClusterECALTimeSelected)



    elif scenario ==2:
        print 'SCENARIO 2: ECAL clustering with time & default ECAL time reconstruction'
        print 'Timing cuts are applied to the PF cluster level'
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitECALWithTime_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterECALTimeSelected_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterECALWithTime_cfi')

        process.particleFlowRecHitECAL = process.particleFlowRecHitECALWithTime.clone()
        process.particleFlowClusterECALUncorrected = process.particleFlowClusterECALWithTimeUncorrected.clone()
        process.particleFlowClusterECALUncorrected.recHitsSource = cms.InputTag("particleFlowRecHitECAL")
        process.particleFlowClusterECALTimeSelected.src = cms.InputTag('particleFlowClusterECALUncorrected')
        process.particleFlowClusterECAL.inputECAL = cms.InputTag('particleFlowClusterECALTimeSelected')

        i = process.pfClusteringECAL.index(process.particleFlowClusterECALUncorrected)
        process.pfClusteringECAL.insert(i+1,process.particleFlowClusterECALTimeSelected)


    elif scenario ==3:
        print 'SCENARIO 3: ECAL clustering with time & ECAL max sample reconstruction using outOfTimeEnergy()'
        print 'Timing cuts are applied to the PF cluster level'
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitECALWithTimeMaxSample_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterECALTimeSelected_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterECALWithTime_cfi')

        process.particleFlowRecHitECAL = process.particleFlowRecHitECALWithTime.clone()
        process.particleFlowClusterECALUncorrected = process.particleFlowClusterECALWithTimeUncorrected.clone()
        process.particleFlowClusterECALUncorrected.recHitsSource = cms.InputTag("particleFlowRecHitECAL")
        process.particleFlowClusterECALTimeSelected.src = cms.InputTag('particleFlowClusterECALUncorrected')
        process.particleFlowClusterECAL.inputECAL = cms.InputTag('particleFlowClusterECALTimeSelected')

        i = process.pfClusteringECAL.index(process.particleFlowClusterECALUncorrected)
        process.pfClusteringECAL.insert(i+1,process.particleFlowClusterECALTimeSelected)

    else:
        print 'UNKNOWN scenario for ECAL. Will keep the default config'
        
        
        
        


        
    
