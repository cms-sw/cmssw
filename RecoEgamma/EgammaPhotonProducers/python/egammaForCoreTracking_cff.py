import FWCore.ParameterSet.Config as cms



from RecoEcal.EgammaClusterProducers.particleFlowSuperClusterECAL_cfi import particleFlowSuperClusterECAL as _particleFlowSuperClusterECAL

#cant use the regression as requires #vertices and we cant use tracking info
#also require it to be above 20 GeV as we only want E/gammas Et>20 and H/E <0.2
particleFlowSuperClusterECALForTrk = _particleFlowSuperClusterECAL.clone(
     useRegression = False,
     regressionConfig = cms.PSet(),
     thresh_SCEt = 20.0
)

egammasForTrk = cms.EDProducer( "EgammaHLTRecoEcalCandidateProducers",
    scIslandEndcapProducer = cms.InputTag( 'particleFlowSuperClusterECALForTrk','particleFlowSuperClusterECALEndcapWithPreshower' ),
    scHybridBarrelProducer = cms.InputTag( 'particleFlowSuperClusterECALForTrk','particleFlowSuperClusterECALBarrel' ),
    recoEcalCandidateCollection = cms.string( "" )
)

egammaHoverEForTrk = cms.EDProducer( "EgammaHLTBcHcalIsolationProducersRegional",
    effectiveAreas = cms.vdouble( 0.105, 0.17 ),
    doRhoCorrection = cms.bool( False ),
    outerCone = cms.double( 0.15 ),
    caloTowerProducer = cms.InputTag( "caloTowerForTrk" ),
    innerCone = cms.double( 0.0 ),
    useSingleTower = cms.bool( False ),
    rhoProducer = cms.InputTag( "" ),
    depth = cms.int32( -1 ),
    absEtaLowEdges = cms.vdouble( 0.0, 1.479 ),
    recoEcalCandidateProducer = cms.InputTag( "egammasForTrk" ),
    rhoMax = cms.double( 9.9999999E7 ),
    etMin = cms.double( 0.0 ),
    rhoScale = cms.double( 1.0 ),
    doEtSum = cms.bool( False )
)

egammasForCoreTracking = cms.EDProducer( "EgammaHLTFilteredEcalCandPtrProducer",
    cands = cms.InputTag( "egammasForTrk" ),
    cuts = cms.VPSet( 
        cms.PSet(
            var = cms.InputTag( "egammaHoverEForTrk" ),
            barrelCut = cms.PSet( 
                useEt = cms.bool( False ),
                cutOverE = cms.double( 0.2 )
            ),
            endcapCut = cms.PSet( 
                useEt = cms.bool( False ),
                cutOverE = cms.double( 0.2 )
            )
        )     
    )                                                        
)                                        
egammaForCoreTrackingTask = cms.Task(particleFlowSuperClusterECALForTrk, 
                                     egammasForTrk,
                                     egammaHoverEForTrk, 
                                     egammasForCoreTracking)

egammaForCoreTrackingSeq = cms.Sequence(egammaForCoreTrackingTask)
