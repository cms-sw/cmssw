import FWCore.ParameterSet.Config as cms

ptRange = cms.vdouble(0 , 13, 30, 70, 1000)
etaRange = cms.vdouble(0 , 1.0, 1.4, 10)

diagTerm = cms.PSet( values = cms.vdouble( 3, 3, 3, 5,
                                           4, 5,  10, 7,
                                           10, 10, 10, 10
                                           ),
                     action = cms.string("scale")
                     )
offDiagTerm = cms.PSet( values = cms.vdouble( 1, 1, 1, 1,
                                              1, 1, 1, 1,
                                              1, 1, 1, 1
                                              ),
                        action = cms.string("scale")
                        )

MuonErrorMatrixValues = cms.PSet(
    errorMatrixPset = cms.PSet(
    action = cms.string('use'),
    atIP = cms.bool(True),
    errorMatrixValuesPSet = cms.PSet(
      xAxis = ptRange,
      yAxis = etaRange,
      zAxis = cms.vdouble(-3.14159, 3.14159),
    
      pf3_V11 = diagTerm,
      pf3_V22 = diagTerm,
      pf3_V33 = diagTerm,
      pf3_V44 = diagTerm,
      pf3_V55 = diagTerm,
      
      pf3_V12 = offDiagTerm,
      pf3_V13 = offDiagTerm,
      pf3_V14 = offDiagTerm,
      pf3_V15 = offDiagTerm,
      pf3_V23 = offDiagTerm,
      pf3_V24 = offDiagTerm,
      pf3_V25 = offDiagTerm,
      pf3_V34 = offDiagTerm,
      pf3_V35 = offDiagTerm,
      pf3_V45 = offDiagTerm
      )
    )
        
    )
