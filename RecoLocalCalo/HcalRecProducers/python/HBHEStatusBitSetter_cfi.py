import FWCore.ParameterSet.Config as cms

qie8Config = cms.PSet(nominalPedestal=cms.double(3.0),  #fC
                      hitEnergyMinimum=cms.double(1.0), #GeV
                      hitMultiplicityThreshold=cms.int32(17),
                      pulseShapeParameterSets = cms.VPSet(
                          cms.PSet(pulseShapeParameters=cms.vdouble(   0.0, 100.0, -50.0, 0.0, -15.0, 0.15)),
                          cms.PSet(pulseShapeParameters=cms.vdouble( 100.0, 2.0e3, -50.0, 0.0,  -5.0, 0.05)),
                          cms.PSet(pulseShapeParameters=cms.vdouble( 2.0e3, 1.0e6, -50.0, 0.0,  95.0, 0.0 )),
                          cms.PSet(pulseShapeParameters=cms.vdouble(-1.0e6, 1.0e6,  45.0, 0.1, 1.0e6, 0.0 )),
                      )
)

# For now, QIE11 parameters duplicate those of QIE8. To be tuned.
qie11Config = cms.PSet(nominalPedestal=cms.double(3.0),  #fC
                      hitEnergyMinimum=cms.double(1.0), #GeV
                      hitMultiplicityThreshold=cms.int32(17),
                      pulseShapeParameterSets = cms.VPSet(
                          cms.PSet(pulseShapeParameters=cms.vdouble(   0.0, 100.0, -50.0, 0.0, -15.0, 0.15)),
                          cms.PSet(pulseShapeParameters=cms.vdouble( 100.0, 2.0e3, -50.0, 0.0,  -5.0, 0.05)),
                          cms.PSet(pulseShapeParameters=cms.vdouble( 2.0e3, 1.0e6, -50.0, 0.0,  95.0, 0.0 )),
                          cms.PSet(pulseShapeParameters=cms.vdouble(-1.0e6, 1.0e6,  45.0, 0.1, 1.0e6, 0.0 )),
                      )
)
