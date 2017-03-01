
import FWCore.ParameterSet.Config as cms



def usePFWithMethodOne(process):
    print '-------------PF with method I -------------'
    print 'Assumes that HCAL reco is tuned to method I '
    process.particleFlowRecHitHBHE.navigator = cms.PSet(
        name = cms.string("PFRecHitHCALNavigator")
    )
    process.pfClusteringHBHEHF.remove(process.particleFlowClusterHBHETimeSelected)
    process.particleFlowClusterHCAL.clustersSource = cms.InputTag("particleFlowClusterHBHE")


def usePFWithMethodOnePointFive(process):
    print '-------------PF with method I.5-------------'
    print 'Independent of which HCAL reco was used since it reads the time samples from the rechit '


    from RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHEMaxSampleTimeSelected_cfi import particleFlowClusterHBHETimeSelected as timeSelector
    process.particleFlowClusterHBHETimeSelected.cuts = timeSelector.cuts

    from RecoParticleFlow.PFClusterProducer.particleFlowCaloResolution_cfi import _timeResolutionHCALMaxSample
    process.particleFlowRecHitHBHE.navigator = cms.PSet(
        name = cms.string("PFRecHitHCALNavigatorWithTime"),
        sigmaCut = cms.double(4.0),
        timeResolutionCalc = _timeResolutionHCALMaxSample
    )

    for p in process.particleFlowRecHitHBHE.producers:
        p.name = cms.string("PFHBHERecHitCreatorMaxSample")
    process.particleFlowClusterHBHE.pfClusterBuilder.algoName = cms.string("PFlow2DClusterizerWithTime")

    


        

