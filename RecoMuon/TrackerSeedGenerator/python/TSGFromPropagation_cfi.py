import FWCore.ParameterSet.Config as cms

SeedGeneratorParameters = cms.PSet(
      ComponentName = cms.string( "TSGFromPropagation" ),
      Propagator = cms.string( "SmartPropagatorAnyOpposite" ),
      MaxChi2 = cms.double( 15.0 ),
      ResetMethod = cms.string("discrete"),
      ErrorRescaling = cms.double(3.0),
      SigmaZ = cms.double(25.0),
      UseVertexState = cms.bool( True ),
      UpdateState = cms.bool( True ),
      SelectState = cms.bool( True ),
      errorMatrixPset = cms.PSet( ),
      UseSecondMeasurements = cms.bool( False )
)
