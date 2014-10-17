
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

        from RecoParticleFlow.PFClusterProducer.particleFlowCaloResolution_cfi import _timeResolutionECALBarrel,_timeResolutionECALEndcap

        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterECALTimeSelected_cfi')

        process.particleFlowRecHitECAL.navigator.name = "PFRecHitECALNavigatorWithTime"
        process.particleFlowRecHitECAL.navigator.barrel = cms.PSet(
             sigmaCut = cms.double(5.0),
             timeResolutionCalc = _timeResolutionECALBarrel 
        )
        process.particleFlowRecHitECAL.navigator.endcap = cms.PSet(
            sigmaCut = cms.double(5.0),
            timeResolutionCalc = _timeResolutionECALEndcap
        )
        
        for p in process.particleFlowRecHitECAL.producers:
            for t in p.qualityTests:
                if t.name == 'PFRecHitQTestECAL':
                    t.timingCleaning = cms.bool(False) 

        process.particleFlowClusterECALUncorrected.recHitsSource = cms.InputTag("particleFlowRecHitECAL")
        process.particleFlowClusterECALTimeSelected.src = cms.InputTag('particleFlowClusterECALUncorrected')
        process.particleFlowClusterECAL.inputECAL = cms.InputTag('particleFlowClusterECALTimeSelected')

        i = process.pfClusteringECAL.index(process.particleFlowClusterECALUncorrected)
        process.pfClusteringECAL.insert(i+1,process.particleFlowClusterECALTimeSelected)
        process.particleFlowClusterECALUncorrected.pfClusterBuilder.algoName = cms.string("PFlow2DClusterizerWithTime")

        process.particleFlowClusterECALUncorrected.pfClusterBuilder.showerSigma = cms.double(1.5)
        process.particleFlowClusterECALUncorrected.pfClusterBuilder.timeSigmaEB = cms.double(10.)
        process.particleFlowClusterECALUncorrected.pfClusterBuilder.timeSigmaEE = cms.double(10.)
        process.particleFlowClusterECALUncorrected.pfClusterBuilder.maxNSigmaTime = cms.double(10.)
        process.particleFlowClusterECALUncorrected.pfClusterBuilder.minChi2Prob = cms.double(0.)
        process.particleFlowClusterECALUncorrected.pfClusterBuilder.clusterTimeResFromSeed = cms.bool(False)
        process.particleFlowClusterECALUncorrected.pfClusterBuilder.timeResolutionCalcBarrel = _timeResolutionECALBarrel
        process.particleFlowClusterECALUncorrected.pfClusterBuilder.timeResolutionCalcEndcap = _timeResolutionECALEndcap

        
        

def customizePFHCALClustering(process,scenario):
    if scenario ==0:
        print '-------------HCAL CLUSTERING-------------'
        print 'Default HCAL reconstruction and no 3D clustering with time in PF'
        print 'but clustering 3D in space from layers'

        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHBHETimeSelected)


        for p in process.particleFlowRecHitHBHE.producers:
            p.name = cms.string('PFHBHERecHitCreator')


        process.particleFlowClusterHCAL.clustersSource = cms.InputTag("particleFlowClusterHBHE")
        process.particleFlowRecHitHBHE.navigator = cms.PSet(
            name = cms.string("PFRecHitHCALNavigator")
        )
        process.particleFlowClusterHBHE.pfClusterBuilder.algoName = cms.string("Basic2DGenericPFlowClusterizer")

