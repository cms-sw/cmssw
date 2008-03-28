import FWCore.ParameterSet.Config as cms

#ES producer for fake ttrig
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.DTGeometry.dtGeometry_cfi import *
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *
DTFakeT0ESProducer = cms.ESSource("DTFakeT0ESProducer",
    t0Mean = cms.double(0.0),
    t0Sigma = cms.double(0.0)
)


