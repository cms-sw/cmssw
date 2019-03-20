import FWCore.ParameterSet.Config as cms

from CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi import *
ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = cms.vstring(
  "Alignment/CTPPS/data/RPixGeometryCorrections.xml"
)
