# from Alignment.MuonAlignment.convertXMLtoSQLite_cfg import *
# process.MuonGeometryDBConverter.fileName = "CSCphotogrammetry.xml"                          # input XML
# process.PoolDBOutputService.connect = "sqlite_file:CSCphotogrammetry.db"                    # output SQLite

# from Alignment.MuonAlignment.convertSQLitetoXML_cfg import *
# process.PoolDBESSource.connect = "sqlite_file:CSCphotogrammetry.db"                         # input SQLite
# process.MuonGeometryDBConverter.outputXML.fileName = "CSCphotogrammetry_reltoideal.xml"     # output XML
# process.MuonGeometryDBConverter.outputXML.relativeto = "ideal"
# process.MuonGeometryDBConverter.outputXML.suppressDTChambers = cms.untracked.bool(True)
# process.MuonGeometryDBConverter.outputXML.suppressDTSuperLayers = cms.untracked.bool(True)
# process.MuonGeometryDBConverter.outputXML.suppressDTLayers = cms.untracked.bool(True)
# process.MuonGeometryDBConverter.outputXML.suppressCSCChambers = cms.untracked.bool(False)
# process.MuonGeometryDBConverter.outputXML.suppressCSCLayers = cms.untracked.bool(True)

from Alignment.MuonAlignment.convertXMLtoXML_cfg import *
process.MuonGeometryDBConverter.fileName = "CSCphotogrammetry.xml"                            # input XML
process.MuonGeometryDBConverter.outputXML.fileName = "CSCphotogrammetry_reltoideal.xml"       # output XML
process.MuonGeometryDBConverter.outputXML.relativeto = "ideal"
process.MuonGeometryDBConverter.outputXML.suppressDTChambers = cms.untracked.bool(True)
process.MuonGeometryDBConverter.outputXML.suppressDTSuperLayers = cms.untracked.bool(True)
process.MuonGeometryDBConverter.outputXML.suppressDTLayers = cms.untracked.bool(True)
process.MuonGeometryDBConverter.outputXML.suppressCSCChambers = cms.untracked.bool(False)
process.MuonGeometryDBConverter.outputXML.suppressCSCLayers = cms.untracked.bool(True)
