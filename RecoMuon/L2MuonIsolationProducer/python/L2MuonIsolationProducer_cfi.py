import FWCore.ParameterSet.Config as cms

L2MuonIsolations = cms.EDProducer("L2MuonIsolationProducer",
    StandAloneCollectionLabel = cms.InputTag("L2Muons","UpdatedAtVtx"),
    IsolatorPSet = cms.PSet( 
      ComponentName = cms.string( "SimpleCutsIsolator" ),
      ConeSizes = cms.vdouble(0.24, 0.24, 0.24, 0.24, 0.24, 
        0.24, 0.24, 0.24, 0.24, 0.24, 
        0.24, 0.24, 0.24, 0.24, 0.24, 
        0.24, 0.24, 0.24, 0.24, 0.24, 
        0.24, 0.24, 0.24, 0.24, 0.24, 
        0.24),
      Thresholds = cms.vdouble(5.5, 5.5, 5.9, 5.7, 5.1, 
        4.9, 5.0, 5.0, 5.1, 5.0, 
        4.8, 4.8, 4.7, 4.7, 3.5, 
        3.1, 3.5, 3.9, 3.7, 3.7, 
        3.5, 3.5, 3.2, 3.3, 3.4, 
        3.4),
      EtaBounds = cms.vdouble(0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 
        0.4785, 0.5655, 0.6525, 0.7395, 0.8265, 
        0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 
        1.3485, 1.4355, 1.5225, 1.6095, 1.6965, 
        1.785, 1.88, 1.9865, 2.1075, 2.247, 
        2.411)
    ),
    WriteIsolatorFloat = cms.bool(False),
#    OutputMuIsoDeposits = cms.bool(True),
    ExtractorPSet = cms.PSet(
        DR_Veto_H = cms.double(0.1),
        Vertex_Constraint_Z = cms.bool(False),
        Threshold_H = cms.double(0.5),
        ComponentName = cms.string('CaloExtractor'),
        Threshold_E = cms.double(0.2),
        DR_Max = cms.double(1.0),
        DR_Veto_E = cms.double(0.07),
        Weight_E = cms.double(1.5),
        Vertex_Constraint_XY = cms.bool(False),
        DepositLabel = cms.untracked.string('EcalPlusHcal'),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        Weight_H = cms.double(1.0)
    )
)



