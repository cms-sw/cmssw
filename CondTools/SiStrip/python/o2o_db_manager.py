import sqlalchemy
import sqlalchemy.ext.declarative
import logging

import CondCore.Utilities.credentials as auth

prod_db_service = ['cms_orcon_prod', 'cms_orcon_prod/cms_cond_general_w']
dev_db_service = ['cms_orcoff_prep', 'cms_orcoff_prep/cms_cond_general_w']
schema_dict = {'cms_orcon_prod':'cms_cond_o2o', 'cms_orcoff_prep':'cms_cond_strip'}
sqlalchemy_tpl = 'oracle://%s:%s@%s'
coral_tpl = 'oracle://%s/%s'
private_db = 'sqlite:///post_o2o.db'
authPathEnvVar = 'COND_AUTH_PATH'

_Base = sqlalchemy.ext.declarative.declarative_base()

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


class DbManager(object):
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

