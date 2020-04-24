import FWCore.ParameterSet.Config as cms

particleFlowClusterHCAL = cms.EDProducer('PFMultiDepthClusterProducer',
       clustersSource = cms.InputTag("particleFlowClusterHBHE"),
       pfClusterBuilder =cms.PSet(
           algoName = cms.string("PFMultiDepthClusterizer"),
           nSigmaEta = cms.double(2.),
           nSigmaPhi = cms.double(2.),
           #pf clustering parameters
           minFractionToKeep = cms.double(1e-7),
           allCellsPositionCalc = cms.PSet(
               algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
               minFractionInCalc = cms.double(1e-9),    
               posCalcNCrystals = cms.int32(-1),
               logWeightDenominator = cms.double(0.8),#same as gathering threshold
               minAllowedNormalization = cms.double(1e-9)
           )
       ),
       positionReCalc = cms.PSet(),
       energyCorrector = cms.PSet()
)
