'''
Deal with the update of config-to-payload map for the fast O2O.

@author: hqu
'''

import os
import sqlalchemy
import sqlalchemy.ext.declarative
import logging
import datetime

import CondCore.Utilities.credentials as auth

prod_db_service = ['cms_orcon_prod', 'cms_orcon_prod/cms_cond_general_w']
dev_db_service = ['cms_orcoff_prep', 'cms_orcoff_prep/cms_test_conditions']
schema_dict = {'cms_orcon_prod':'CMS_COND_O2O', 'cms_orcoff_prep':'CMS_COND_STRIP'}
sqlalchemy_tpl = 'oracle://%s:%s@%s'
coral_tpl = 'oracle://%s/%s'
private_db = 'sqlite:///post_o2o.db'
authPathEnvVar = 'COND_AUTH_PATH'

_Base = sqlalchemy.ext.declarative.declarative_base()

class ConfigToPayloadMapDef(object):
    __tablename__ = 'STRIP_CONFIG_TO_PAYLOAD_MAP'
    config_hash = sqlalchemy.Column(sqlalchemy.String(2000), primary_key=True)
    payload_hash = sqlalchemy.Column(sqlalchemy.String(2000), nullable=False)
    payload_type = sqlalchemy.Column(sqlalchemy.String(2000), nullable=False)
    config_string = sqlalchemy.Column(sqlalchemy.String(2000), nullable=False)
    insertion_time = sqlalchemy.Column(sqlalchemy.TIMESTAMP, nullable=False)

class GainLogsDef(object):
    __tablename__ = 'STRIP_TICKMARK_GAIN_LOG'
    iov = sqlalchemy.Column(sqlalchemy.BIGINT, primary_key=True)
    insertion_time = sqlalchemy.Column(sqlalchemy.TIMESTAMP, primary_key=True)
    config_string = sqlalchemy.Column(sqlalchemy.String(2000), nullable=False)
    skipped_detids = sqlalchemy.Column(sqlalchemy.CLOB, nullable=True)
    description = sqlalchemy.Column(sqlalchemy.CLOB, nullable=True)

def make_dbtype(base_class, schema=None):
    import re
    name = re.sub('Def$', '', base_class.__name__)
    members = {}
    members['__tablename__'] = base_class.__tablename__
    members['__table_args__'] = None
    if schema:
        name = name + schema
        members['__table_args__'] = {'schema' : schema}
    dbType = type(name, (_Base, base_class), members)
    return dbType


class O2OFinalizer(object):
    def __init__(self, db, authFile=None):
        self.authFile = authFile
        if db == 'prod':
            self.db_service = prod_db_service
        elif db == 'dev':
            self.db_service = dev_db_service
        elif db == 'private':
            self.db_service = None
        else:
            raise RuntimeError('Option db(=%s) is not in the supported database list: [prod, dev, private]' % db)

        logging.info('Connecting to %s database' % self.db_service[0] if self.db_service else private_db)

        self.schema = schema_dict[self.db_service[0]] if self.db_service else None
        self.ConfigToPayloadMap = make_dbtype(ConfigToPayloadMapDef, self.schema)
        self.GainLogs = make_dbtype(GainLogsDef, self.schema)
        if self.schema:
            self.ConfigToPayloadMapSqlite = make_dbtype(ConfigToPayloadMapDef, schema=None)
        else:
            self.ConfigToPayloadMapSqlite = self.ConfigToPayloadMap

    def get_url(self, force_schema=False):
        if self.db_service == None:
            url = private_db
        else:
            authEntry = self.db_service[1]
            if force_schema and self.schema:
                authEntry = '%s/%s' % (self.db_service[0], self.schema)
            username, _, pwd = auth.get_credentials(authPathEnvVar, authEntry, self.authFile)
            url = sqlalchemy_tpl % (username, pwd, self.db_service[0])
        return url

    def check_table(self, table_def, table_class):
        self.engine = sqlalchemy.create_engine(self.get_url())
        if not self.engine.has_table(table_def.__tablename__, self.schema):
            logging.info('Creating table %s on %s' % (table_def.__tablename__,
                                                      self.db_service[0] if self.db_service else private_db))
            self.engine = sqlalchemy.create_engine(self.get_url(True))
            table_class.__table__.create(bind=self.engine)
        self.session = sqlalchemy.orm.scoped_session(sqlalchemy.orm.sessionmaker(bind=self.engine))

    def connect(self, url=None):
        engine = sqlalchemy.create_engine(url) if url else self.engine
        session = sqlalchemy.orm.scoped_session(sqlalchemy.orm.sessionmaker(bind=engine))
        return session

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

    def update_gain_logs(self, iov, cfglines, skipfile, descfile):
        skipped = ''
        description = ''

        if os.path.exists(skipfile):
            with open(skipfile) as f:
                skipped = f.read()
        else:
            logging.warning('Skipped module file %s does not exist!' % skipfile)

        if os.path.exists(descfile):
            with open(descfile) as f:
                description = f.read()
        else:
            logging.warning('Description file %s does not exist!' % descfile)

        self.check_table(GainLogsDef, self.GainLogs)
        destSession = self.connect()
        entry = self.GainLogs(iov=iov, 
                              insertion_time=datetime.datetime.now(),
                              config_string=cfglines,
                              skipped_detids=skipped, 
                              description=description)
        destSession.add(entry)
        destSession.commit()
        logging.info('Info on the skipped modules saved in database:\n' + description)
