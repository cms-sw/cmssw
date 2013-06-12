import os

execfile(os.environ["CMSSW_RELEASE_BASE"] + "/src/Alignment/CommonAlignmentProducer/test/GlobalPositionRcd-write_cfg.py")

process.PoolDBOutputService.connect = "sqlite_file:inertGlobalPositionRcd.db"
process.PoolDBOutputService.toPut[0].tag = "inertGlobalPositionRcd"
