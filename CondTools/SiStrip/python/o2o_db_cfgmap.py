'''
Update the config-to-payload map table for the fast DAQ O2O.

@author: hqu
'''

from CondTools.SiStrip.o2o_db_manager import make_dbtype, DbManager
import os
import logging
import sqlalchemy


class ConfigToPayloadMapDef(object):
    __tablename__ = 'STRIP_CONFIG_TO_PAYLOAD_MAP'
    config_hash = sqlalchemy.Column(sqlalchemy.String(2000), primary_key=True)
    payload_hash = sqlalchemy.Column(sqlalchemy.String(2000), nullable=False)
    payload_type = sqlalchemy.Column(sqlalchemy.String(2000), nullable=False)
    config_string = sqlalchemy.Column(sqlalchemy.String(2000), nullable=False)
    insertion_time = sqlalchemy.Column(sqlalchemy.TIMESTAMP, nullable=False)


class DbManagerDAQ(DbManager):
    def __init__(self, db, authPath=None):
        DbManager.__init__(self, db, authPath)
        self.ConfigToPayloadMap = make_dbtype(ConfigToPayloadMapDef, self.schema)
        if self.schema:
            self.ConfigToPayloadMapSqlite = make_dbtype(ConfigToPayloadMapDef, schema=None)
        else:
            self.ConfigToPayloadMapSqlite = self.ConfigToPayloadMap

    def update_hashmap(self, input_path):
        if not os.path.exists(input_path):
            logging.info('No config-to-payload map file at %s. Skipping.' % input_path)
            return
        session = self.connect('sqlite:///%s' % input_path)
        entry = session.query(self.ConfigToPayloadMapSqlite).first()
        if entry:
            self.check_table(ConfigToPayloadMapDef, self.ConfigToPayloadMap)
            destSession = self.connect()
            cfgmap = self.ConfigToPayloadMap(config_hash=entry.config_hash,
                                             payload_hash=entry.payload_hash,
                                             payload_type=entry.payload_type,
                                             config_string=entry.config_string,
                                             insertion_time=entry.insertion_time)
            destSession.add(cfgmap)
            destSession.commit()
            logging.info('Updated config-to-payload map for %s' % cfgmap.payload_type)
            logging.info('... config_hash = %s, payload_hash = %s' % (cfgmap.config_hash, cfgmap.payload_hash))
        else:
            raise RuntimeError('No entry found in config-to-payload map file %s' % input_path)
