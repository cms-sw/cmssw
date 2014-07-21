import FWCore.ParameterSet.Config as cms





def customizePFECALClustering(process,scenario):

    if scenario ==1:
        print '-------------ECAL CLUSTERING-------------'
        print 'SCENARIO 1: Default ECAL clustering & ECAL time reconstruction'
        print 'Timing cuts are applied to the PF cluster level'
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterECALTimeSelected_cfi')
        process.particleFlowClusterECALTimeSelected.src = cms.InputTag('particleFlowClusterECALUncorrected')
        process.particleFlowClusterECAL.inputECAL = cms.InputTag('particleFlowClusterECALTimeSelected')
        
        i = process.pfClusteringECAL.index(process.particleFlowClusterECALUncorrected)
        process.pfClusteringECAL.insert(i+1,process.particleFlowClusterECALTimeSelected)



    elif scenario ==2:
        print '-------------ECAL CLUSTERING-------------'
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
        print '-------------ECAL CLUSTERING-------------'
        print 'SCENARIO 3: ECAL clustering with time & ECAL max sample reconstruction using outOfTimeEnergy()'
        print 'Timing cuts are applied to the PF cluster level'
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitECALWithTime_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterECALTimeSelected_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterECALWithTime_cfi')

        process.particleFlowRecHitECAL = process.particleFlowRecHitECALWithTime.clone()
        for p in process.particleFlowRecHitECAL.producers:
            p.name=cms.string(p.name.value()+'MaxSample')
        process.particleFlowClusterECALUncorrected = process.particleFlowClusterECALWithTimeUncorrected.clone()
        process.particleFlowClusterECALUncorrected.recHitsSource = cms.InputTag("particleFlowRecHitECAL")
        process.particleFlowClusterECALTimeSelected.src = cms.InputTag('particleFlowClusterECALUncorrected')
        process.particleFlowClusterECAL.inputECAL = cms.InputTag('particleFlowClusterECALTimeSelected')

        i = process.pfClusteringECAL.index(process.particleFlowClusterECALUncorrected)
        process.pfClusteringECAL.insert(i+1,process.particleFlowClusterECALTimeSelected)



        
        
        
        

def customizePFHCALClustering(process,scenario):
    if scenario ==1:
        print '-------------HCAL CLUSTERING-------------'
        print 'SCENARIO 1: HCAL clustering from hits'
        print 'No timing cuts applied'
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHE_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitHF_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHE_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterHF_cfi')

        process.pfClusteringHBHEHF.remove(process.towerMakerPF)
        process.pfClusteringHBHEHF.remove(process.particleFlowRecHitHCAL)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHCAL)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHFHAD)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHFEM)

        process.pfClusteringHBHEHF+=process.particleFlowRecHitHBHE
        process.pfClusteringHBHEHF+=process.particleFlowRecHitHF
        process.pfClusteringHBHEHF+=process.particleFlowClusterHBHE
        process.pfClusteringHBHEHF+=process.particleFlowClusterHF

    elif scenario ==2:
        from RecoParticleFlow.PFClusterProducer.particleFlowCaloResolution_cfi import _timeResolutionHCAL
        print '-------------HCAL CLUSTERING-------------'
        print 'SCENARIO 2: HCAL clustering with time using the maximum time sample '
        print 'No timing cuts applied'
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHE_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitHF_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHEWithTime_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterHF_cfi')

        process.pfClusteringHBHEHF.remove(process.towerMakerPF)
        process.pfClusteringHBHEHF.remove(process.particleFlowRecHitHCAL)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHCAL)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHFHAD)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHFEM)
        

        process.particleFlowClusterHBHE=process.particleFlowClusterHBHEWithTime.clone()
        
        process.pfClusteringHBHEHF+=process.particleFlowRecHitHBHE
        process.pfClusteringHBHEHF+=process.particleFlowRecHitHF
        process.pfClusteringHBHEHF+=process.particleFlowClusterHBHE
        process.pfClusteringHBHEHF+=process.particleFlowClusterHF

        process.particleFlowRecHitHBHE.navigator = cms.PSet(
            name = cms.string("PFRecHitHCALNavigatorWithTime"),
            sigmaCut = cms.double(1),
            timeResolutionCalc = _timeResolutionHCAL
        )
    
        for p in process.particleFlowRecHitHBHE.producers:
            p.name=cms.string(p.name.value()+'MaxSample')
            for q in p.qualityTests:
                if q.name.value() =="PFRecHitQTestThreshold":
                    print 'lowering rechit threhold to 0.25 GeV since we are using 1 time sample'
                    q.threshold = cms.double(0.25)
                    
