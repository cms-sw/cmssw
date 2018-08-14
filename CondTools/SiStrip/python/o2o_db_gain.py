'''
Update the bookkeeping tables for G1 O2O.

@author: hqu
'''

from CondTools.SiStrip.o2o_db_manager import make_dbtype, DbManager
import os
import sys
import datetime
import logging
from importlib import import_module
import sqlalchemy
from sqlalchemy.ext.declarative import declared_attr

class GainO2OPartitionDef(object):
    __tablename__ = 'STRIP_GAIN_O2O_PARTITION'
    o2oid = sqlalchemy.Column(sqlalchemy.BigInteger, primary_key=True)
    iovstart = sqlalchemy.Column(sqlalchemy.BigInteger, nullable=False)
    o2otimestamp = sqlalchemy.Column(sqlalchemy.TIMESTAMP, nullable=False)
    subDetector = sqlalchemy.Column(sqlalchemy.String(256), primary_key=True)
    partitionname = sqlalchemy.Column(sqlalchemy.String(256), nullable=False)
    fecVersionMajorId = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    fecVersionMinorId = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    fedVersionMajorId = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    fedVersionMinorId = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    connectionVersionMajorId = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    connectionVersionMinorId = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    dcuInfoVersionMajorId = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    dcuInfoVersionMinorId = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    dcuPsuMapVersionMajorId = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    dcuPsuMapVersionMinorId = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    maskVersionMajorId = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    maskVersionMinorId = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    apvTimingVersionMajorId = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    apvTimingVersionMinorId = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    timingRunNumber = sqlalchemy.Column(sqlalchemy.BigInteger, nullable=False)

class GainO2OSkippedDevicesDef(object):
    __tablename__ = 'STRIP_GAIN_O2O_SKIPPED'
    # o2o identifier
    o2oid = sqlalchemy.Column(sqlalchemy.BigInteger, primary_key=True)
    itemid = sqlalchemy.Column(sqlalchemy.BigInteger, primary_key=True)
    # FEC coordinates
    fecCrate = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    fecSlot = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    fecRing = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    ccuAddr = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    ccuChan = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    lldChan = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    i2cAddr = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    # FED coordinates
    fedId = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    feUnit = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    feChan = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    fedApv = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    # detid
    detid = sqlalchemy.Column(sqlalchemy.BigInteger, nullable=True)
    
class GainO2OWhitelistedDevicesDef(GainO2OSkippedDevicesDef):
    __tablename__ = 'STRIP_GAIN_O2O_WHITELISTED'


class DbManagerGain(DbManager):
    def __init__(self, db, authFile=None):
        DbManager.__init__(self, db, authFile)
        self.GainO2OPartition = make_dbtype(GainO2OPartitionDef, self.schema)
        self.GainO2OSkippedDevices = make_dbtype(GainO2OSkippedDevicesDef, self.schema)
        self.GainO2OWhitelistedDevices = make_dbtype(GainO2OWhitelistedDevicesDef, self.schema)

    def _readPartitions(self, p):
        self.o2o_partitions = []
        partitionDict = {'PartTECM':'TEC-','PartTECP':'TEC+','PartTIBD':'TIB/TID','PartTOB':'TOB'}
        o2otimestamp = datetime.datetime.utcnow()
        for part in partitionDict:
            psetPart = p.SiStripConfigDb.Partitions.getParameter(part)
            if not psetPart: continue
            entry = self.GainO2OPartition(
                o2oid = self.o2oid,
                iovstart = self.iovstart,
                o2otimestamp = o2otimestamp,
                subDetector = partitionDict[part],
                partitionname = psetPart.getParameter('PartitionName').value(),
                fecVersionMajorId = psetPart.getParameter('FecVersion')[0],
                fecVersionMinorId = psetPart.getParameter('FecVersion')[1],
                fedVersionMajorId = psetPart.getParameter('FedVersion')[0],
                fedVersionMinorId = psetPart.getParameter('FedVersion')[1],
                connectionVersionMajorId = psetPart.getParameter('CablingVersion')[0],
                connectionVersionMinorId = psetPart.getParameter('CablingVersion')[1],
                dcuInfoVersionMajorId = psetPart.getParameter('DcuDetIdsVersion')[0],
                dcuInfoVersionMinorId = psetPart.getParameter('DcuDetIdsVersion')[1],
                dcuPsuMapVersionMajorId = psetPart.getParameter('DcuPsuMapVersion')[0],
                dcuPsuMapVersionMinorId = psetPart.getParameter('DcuPsuMapVersion')[1],
                maskVersionMajorId = psetPart.getParameter('MaskVersion')[0],
                maskVersionMinorId = psetPart.getParameter('MaskVersion')[1],
                apvTimingVersionMajorId = psetPart.getParameter('ApvTimingVersion')[0],
                apvTimingVersionMinorId = psetPart.getParameter('ApvTimingVersion')[1],
                timingRunNumber = psetPart.getParameter('RunNumber').value()
                )
            self.o2o_partitions.append(entry)

    def _readSkippedDevices(self, p, whitelist=False):
        dev_type = self.GainO2OSkippedDevices
        attr_name = 'SkippedDevices'
        if whitelist:
            dev_type = self.GainO2OWhitelistedDevices
            attr_name = 'WhitelistedDevices'

        dev_list = []
        value = lambda p: None if p is None else p.value()
        for itemid, pset in enumerate(getattr(p.SiStripCondObjBuilderFromDb, attr_name)):
            entry = dev_type(
                o2oid = self.o2oid,
                itemid = itemid,
                fecCrate = value(pset.getParameter('fecCrate')),
                fecSlot = value(pset.getParameter('fecSlot')),
                fecRing = value(pset.getParameter('fecRing')),
                ccuAddr = value(pset.getParameter('ccuAddr')),
                ccuChan = value(pset.getParameter('ccuChan')),
                lldChan = value(pset.getParameter('lldChan')),
                i2cAddr = value(pset.getParameter('i2cAddr')),
                fedId = value(pset.getParameter('fedId')),
                feUnit = value(pset.getParameter('feUnit')),
                feChan = value(pset.getParameter('feChan')),
                fedApv = value(pset.getParameter('fedApv')),
                detid = value(pset.getParameter('detid'))
                )
            dev_list.append(entry)

        return dev_list

    def update_gain_logs(self, iov, cfgname):
        """Insert bookkeeping info to the tables.
        Args:
            iov (int or str): IOV number
            cfgname (str): name of the CMSSW cfg file.
                The cfg file need to be placed in the current directory.
        """

        self.iovstart = int(iov)
        sys.path.append(os.getcwd())
        if cfgname.endswith('.py'):
            cfgname = cfgname.replace('.py', '')
        cfg = import_module(cfgname)

        self.check_table(GainO2OPartitionDef, self.GainO2OPartition)
        self.check_table(GainO2OSkippedDevicesDef, self.GainO2OSkippedDevices)
        self.check_table(GainO2OWhitelistedDevicesDef, self.GainO2OWhitelistedDevices)
        destSession = self.connect()
        o2oid = destSession.query(self.GainO2OPartition.o2oid).order_by(self.GainO2OPartition.o2oid.desc()).first()
        if o2oid:
            self.o2oid = o2oid[0] + 1
        else:
            self.o2oid = 0

        self._readPartitions(cfg.process)
        for entry in self.o2o_partitions:
            destSession.add(entry)

        o2o_skipped = self._readSkippedDevices(cfg.process)
        for entry in o2o_skipped:
            destSession.add(entry)

        o2o_whitelisted = self._readSkippedDevices(cfg.process, whitelist=True)
        for entry in o2o_whitelisted:
            destSession.add(entry)

        destSession.commit()
        logging.info('Inserted Gain O2O info to DB!')


if __name__ == '__main__':
    # for testing
    dbmgr = DbManagerGain('private')
    dbmgr.update_gain_logs(1, 'test')
