import FWCore.ParameterSet.Config as cms

process = cms.Process("COMPUTEIDEAL")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")

# Ideal records 
# frontier://FrontierProd/CMS_COND_31X_FROM21X     DTAlignmentRcd            DTIdealGeometry200_mc
# frontier://FrontierProd/CMS_COND_31X_FROM21X     DTAlignmentErrorExtendedRcd       DTIdealGeometryErrors200_mc
# frontier://FrontierProd/CMS_COND_31X_ALIGNMENT   CSCAlignmentRcd           CSCIdealGeometry310me42_mc
# frontier://FrontierProd/CMS_COND_31X_ALIGNMENT   CSCAlignmentErrorExtendedRcd      CSCIdealGeometryErrors310me42_mc
# frontier://FrontierProd/CMS_COND_31X_FROM21X     GlobalPositionRcd         IdealGeometry
# frontier://FrontierProd/CMS_COND_31X_FROM21X     TrackerAlignmentRcd       TrackerIdealGeometry210_mc
# frontier://FrontierProd/CMS_COND_31X_FROM21X     TrackerAlignmentErrorExtendedRcd  TrackerIdealGeometryErrors210_mc

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.ideal31Xfrom21X = cms.ESSource("PoolDBESSource",
                                       process.CondDBSetup,
                                       connect = cms.string("frontier://FrontierProd/CMS_COND_31X_FROM21X"),
                                       toGet = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTIdealGeometry200_mc")),
                                                         cms.PSet(record = cms.string("DTAlignmentErrorExtendedRcd"), tag = cms.string("DTIdealGeometryErrors200_mc")),
                                                         cms.PSet(record = cms.string("GlobalPositionRcd"), tag = cms.string("IdealGeometry"))))
process.ideal31X = cms.ESSource("PoolDBESSource",
                                process.CondDBSetup,
                                connect = cms.string("frontier://FrontierProd/CMS_COND_31X_ALIGNMENT"),
                                toGet = cms.VPSet(cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCIdealGeometry310me42_mc")),
                                                  cms.PSet(record = cms.string("CSCAlignmentErrorExtendedRcd"), tag = cms.string("CSCIdealGeometryErrors310me42_mc"))))

process.ComputeTransformation = cms.EDAnalyzer("ComputeTransformation", fileName = cms.string("Alignment/MuonAlignment/data/idealTransformation.py"))
process.Path = cms.Path(process.ComputeTransformation)
