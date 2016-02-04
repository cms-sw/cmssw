from Alignment.MuonAlignment.convertXMLtoSQLite_cfg import *
process.MuonGeometryDBConverter.fileName = "Alignment/MuonAlignmentAlgorithms/test/APE1000cm.xml"
process.PoolDBOutputService.connect = "sqlite_file:APE1000cm.db"
