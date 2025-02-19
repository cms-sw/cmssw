import os

fileName = os.getenv("ALIGNMENT_CONVERTXML")

from Alignment.MuonAlignment.convertSQLitetoXML_cfg import *
process.PoolDBESSource.connect = "sqlite_file:%s" % fileName
process.MuonGeometryDBConverter.outputXML.fileName = "%s_global.xml" % fileName
process.MuonGeometryDBConverter.outputXML.relativeto = "none"
process.MuonGeometryDBConverter.outputXML.suppressDTChambers = True
process.MuonGeometryDBConverter.outputXML.suppressDTSuperLayers = True
process.MuonGeometryDBConverter.outputXML.suppressDTLayers = True
process.MuonGeometryDBConverter.outputXML.suppressCSCChambers = False
process.MuonGeometryDBConverter.outputXML.suppressCSCLayers = True

process.MuonGeometryDBConverter.getAPEs = cms.bool(False)
process.PoolDBESSource.toGet = cms.VPSet(
    cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")),
    cms.PSet(record = cms.string("DTAlignmentErrorRcd"), tag = cms.string("DTAlignmentErrorRcd")),
    cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")),
    cms.PSet(record = cms.string("CSCAlignmentErrorRcd"), tag = cms.string("CSCAlignmentErrorRcd")),
      )
