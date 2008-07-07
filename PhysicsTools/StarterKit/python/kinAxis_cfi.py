import FWCore.ParameterSet.Config as cms


def kinAxis(apt1, apt2, am1, am2) :
    return cms.ParameterSet(
        pt1 = cms.double( apt1 ),
        pt2 = cms.double( apt2 ),
        m1 = cms.double( am1 ),
        m2 = cms.double( am2 )
        )
