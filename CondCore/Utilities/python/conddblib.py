'''CMS Conditions DB Python library.
'''

__author__ = 'Miguel Ojeda'
__copyright__ = 'Copyright 2013, CERN'
__credits__ = ['Giacomo Govi', 'Miguel Ojeda', 'Andreas Pfeiffer']
__license__ = 'Unknown'
__maintainer__ = 'Miguel Ojeda'
__email__ = 'mojedasa@cern.ch'


import os
import hashlib
import logging

import sqlalchemy
import sqlalchemy.ext.declarative

authPathEnvVar = 'COND_AUTH_PATH'
schema_name = 'CMS_CONDITIONS'
dbuser_name = 'cms_conditions'
dbreader_user_name = 'cms_cond_general_r'
dbwriter_user_name = 'cms_cond_general_w'
devdbwriter_user_name = 'cms_test_conditions'
logger = logging.getLogger(__name__)

# Set initial level to WARN.  This so that log statements don't occur in
# the absense of explicit logging being enabled.
if logger.level == logging.NOTSET:
    logger.setLevel(logging.WARN)


class EnumMetaclass(type):
    def __init__(cls, name, bases, dct):
        cls._members = sorted([member for member in dir(cls) if not member.startswith('_')])
        cls._map = dict([(member, getattr(cls, member)) for member in cls._members])
        cls._reversemap = dict([(value, key) for (key, value) in cls._map.items()])
        super(EnumMetaclass, cls).__init__(name, bases, dct)

    def __len__(cls):
        return len(cls._members)

    def __getitem__(cls, key):
        '''Returns the value for this key (if the key is an integer,
        the value is the nth member from the sorted members list).
        '''

        if isinstance(key, int):
            # for tuple() and list()
            key = cls._members[key]
        return cls._map[key]

    def __call__(cls, value):
        '''Returns the key for this value.
        '''

        return cls._reversemap[value]


class Enum(object):
    '''A la PEP 435, simplified.
    '''

    __metaclass__ = EnumMetaclass


# Utility functions
def hash(data):
    return hashlib.sha1(data).hexdigest()


# Constants
empty_label = '-'

name_length = 100
description_length = 4000
hash_length = len(hash(''))

web_experts_email = 'cms-cond-dev@cern.ch'
offline_db_experts_email = 'cms-offlinedb-exp@cern.ch'
offline_db_experts_phone = '+41 22 76 70817, or 70817 from CERN; check https://twiki.cern.ch/twiki/bin/viewauth/CMS/DBShifterHelpPage if it does not work; availability depends on the state of the LHC'

contact_help = 'If you need assistance, please write an email to %s and %s. If you need immediate/urgent assistance, you can call the Offline DB expert on call (%s).' % (offline_db_experts_email, web_experts_email, offline_db_experts_phone)
database_help = '''
    The database parameter (--db) refers to the database where the tool
    will connect to read all the data. By default, the production account
    (through Frontier) will be used.

    In subcommands which take a source and a destination, --db always refers to
    the source, and --destdb to the destination. For both of them the following
    rules apply.

    The database parameter can be an official alias, a filename or any
    valid SQLAlchemy URL.

    The official aliases are the following strings (first column):

      Alias         Level        Database       RO/RW       Notes
      ------------  -----------  -------------  ----------  -------------------------------

      pro           Production   Frontier (ADG) read-only   Default.
      arc           Archive      Frontier       read-only
      int           Integration  Frontier       read-only
      dev           Development  Frontier       read-only
      boost         Production   Frontier       read-only
      boostprep     Development  Frontier       read-only

      orapro        Production   Oracle (ADG)   read-only   Password required.
      oraarc        Archive      Oracle         read-only   Password required.
      oraint        Integration  Oracle         read-write  Password required.
      oradev        Development  Oracle         read-write  Password required.

      onlineorapro  Production   Oracle         read-write  Password required. Online only.
      onlineoraint  Online Int   Oracle         read-write  Password required. Online only.

    Most of the time, if you are a regular user, you will want to read/copy
    conditions from the Frontier production account. Therefore, you can omit
    the --db parameter, unless you want to read from somewhere else,
    e.g. from your local SQLite file.

    In addition, the parameter may be a filename (path) pointing to a local
    SQLite file, e.g.

      file.db
      relative/path/to/file.db
      /absolute/path/to/file.db

    Finally, any valid SQLAlchemy URL can be used. This allows full
    flexibility in cases where it may be needed, e.g.

      sqlite://              In-memory, volatile SQLite DB.
      oracle://user@devdb11  Your private Oracle DB in devdb11 [*]

        [*] See https://account.cern.ch/ -> Services for more information
            on personal Oracle accounts.

    For the official aliases, the password will be asked automatically
    interactively. The same applies for Oracle URLs where the password
    was not provided inside it, e.g.:

      oracle://user@devdb11       The tool will prompt you for the password.
      oracle://user:pass@devdb11  Password inlined. [+]

        [+] Caution: Never write passwords in command-line parameters in
            multi-user machines (e.g. lxplus), since other users can see them
            in the process table (e.g. ps).

    This means that both the official aliases and the filenames are shortcuts
    to the full SQLAlchemy URL equivalents, e.g. the following are the same:

       relative/path/to/file.db  ===  sqlite:///relative/path/to/file.db
      /absolute/path/to/file.db  ===  sqlite:////absolute/path/to/file.db
'''

class Synchronization(Enum):
    any        = 'any'
    validation = 'validation'
    mc         = 'mc'
    runmc      = 'runmc'
    hlt        = 'hlt'
    express    = 'express'
    prompt     = 'prompt'
    pcl        = 'pcl'
    offline    = 'offline'

class TimeType(Enum):
    run  = 'Run'
    time = 'Time'
    lumi = 'Lumi'
    hash = 'Hash'
    user = 'User'


# Schema definition
_Base = sqlalchemy.ext.declarative.declarative_base()

def fq_name( schema_name, table_name ):
    name = table_name
    if schema_name is not None:
        name = '%s.%s' %(schema_name, table_name)
    return name

db_models = {}
ora_types = {}
sqlite_types = {} 

class _Col(Enum):
    nullable = 0
    notNull = 1
    pk = 2

class DbRef:
    def __init__(self,refType, refColumn):
        self.rtype = refType
        self.rcol = refColumn   

def fq_col( schema, table, column ):
    fqn = '%s.%s' %(table, column)
    if schema is not None:
        fqn = '%s.%s' %(schema,fqn)
    return fqn

def make_dbtype( backendName, schemaName, baseType ):
    members = {}
    deps_reg = set()
    dbtype_name = '%s_%s' %(baseType.__name__,backendName)
    members['__tablename__'] = baseType.__tablename__
    members['__table_args__'] = None
    if schemaName is not None:
        members['__table_args__'] = {'schema': schemaName }
    for k,v in baseType.columns.items():
        if isinstance(v[0],DbRef):
            refColDbt = v[0].rtype.columns[v[0].rcol][0]
            pk = (True if v[1]==_Col.pk else False)
            if v[1]==_Col.pk:
                members[k] = sqlalchemy.Column(refColDbt,sqlalchemy.ForeignKey(fq_col(schemaName,v[0].rtype.__tablename__,v[0].rcol)),primary_key=True)
            else:
                nullable = (False if v[1] == _Col.notNull else True)
                members[k] = sqlalchemy.Column(refColDbt,sqlalchemy.ForeignKey(fq_col(schemaName,v[0].rtype.__tablename__,v[0].rcol)),nullable=nullable)
            if v[0].rtype.__name__ not in deps_reg:
                deps_reg.add(v[0].rtype.__name__)
                reftype_name = '%s_%s' %(v[0].rtype.__name__,backendName)
                members[(v[0].rtype.__name__).lower()] = sqlalchemy.orm.relationship(reftype_name)
        else:
            if v[1]==_Col.pk:
                members[k] = sqlalchemy.Column(v[0],primary_key=True)
            else:
                nullable = (True if v[1]==_Col.nullable else False)
                members[k] = sqlalchemy.Column(v[0],nullable=nullable)
    dbType = type(dbtype_name,(_Base,),members)
    
    if backendName not in db_models.keys():
        db_models[backendName] = {}
    db_models[backendName][baseType.__name__] = dbType
    return dbType

def getSchema(tp):
    if tp.__table_args__ is not None:
        return tp.__table_args__['schema']
    return None

# notice: the GT table names are _LOWERCASE_. When turned to uppercase, the sqlalchemy ORM queries on GLOBAL_TAG and GLOBAL_TAG_MAP
# dont work ( probably clashes with a GLOBAL keyword in their code?  

class Tag:
    __tablename__       = 'TAG'
    columns             = { 'name': (sqlalchemy.String(name_length),_Col.pk), 
                            'time_type': (sqlalchemy.Enum(*tuple(TimeType)),_Col.notNull),
                            'object_type': (sqlalchemy.String(name_length),_Col.notNull),
                            'synchronization': (sqlalchemy.Enum(*tuple(Synchronization)),_Col.notNull),
                            'description': (sqlalchemy.String(description_length),_Col.notNull),
                            'last_validated_time':(sqlalchemy.BIGINT,_Col.notNull),
                            'end_of_validity':(sqlalchemy.BIGINT,_Col.notNull),
                            'insertion_time':(sqlalchemy.TIMESTAMP,_Col.notNull),
                            'modification_time':(sqlalchemy.TIMESTAMP,_Col.notNull) }


class Payload:
    __tablename__       = 'PAYLOAD'
    columns             = { 'hash': (sqlalchemy.CHAR(hash_length),_Col.pk),
                            'object_type': (sqlalchemy.String(name_length),_Col.notNull),
                            'data': (sqlalchemy.BLOB,_Col.notNull),
                            'streamer_info':(sqlalchemy.BLOB,_Col.notNull),
                            'version':(sqlalchemy.String(20),_Col.notNull),
                            'insertion_time':(sqlalchemy.TIMESTAMP,_Col.notNull) }


class IOV:
    __tablename__       = 'IOV'
    columns             = { 'tag_name':(DbRef(Tag,'name'),_Col.pk),    
                            'since':(sqlalchemy.BIGINT,_Col.pk),
                            'insertion_time':(sqlalchemy.TIMESTAMP,_Col.pk),
                            'payload_hash':(DbRef(Payload,'hash'),_Col.pk) }


# the string  'GLOBAL' being a keyword in sqlalchemy ( upper case ), when used in the model cause the two GT tables to be unreadable ( bug ) 
# the only choice is to use lower case names, and rename the tables in sqlite after creation!!
class GlobalTag:
    __tablename__       = 'global_tag'
    columns             = { 'name':(sqlalchemy.String(name_length),_Col.pk),
                            'validity': (sqlalchemy.BIGINT,_Col.notNull),
                            'description':(sqlalchemy.String(description_length),_Col.notNull),
                            'release':(sqlalchemy.String(name_length),_Col.notNull),
                            'insertion_time':(sqlalchemy.TIMESTAMP,_Col.notNull),
                            'snapshot_time':(sqlalchemy.TIMESTAMP,_Col.notNull) }

class GlobalTagMap:
    __tablename__       = 'global_tag_map'
    columns             = { 'global_tag_name':(DbRef(GlobalTag,'name'),_Col.pk),
                            'record':(sqlalchemy.String(name_length),_Col.pk),
                            'label':(sqlalchemy.String(name_length),_Col.pk),
                            'tag_name':(DbRef(Tag,'name'),_Col.notNull) }



class TagLog:
    __tablename__       = 'TAG_LOG'
    columns             = { 'tag_name':(DbRef(Tag,'name'),_Col.pk),
                            'event_time':(sqlalchemy.TIMESTAMP,_Col.pk), 
                            'action':(sqlalchemy.String(100),_Col.pk),
                            'user_name':(sqlalchemy.String(100),_Col.notNull),
                            'host_name':(sqlalchemy.String(100),_Col.notNull),
                            'command':(sqlalchemy.String(500),_Col.notNull),
                            'user_text':(sqlalchemy.String(4000),_Col.notNull) }


# CondDB object
class Connection(object):

    def __init__(self, url, init=False):
        # Workaround to avoid creating files if not present.
        # Python's sqlite3 module does not use sqlite3_open_v2(),
        # and therefore we cannot disable SQLITE_OPEN_CREATE.
        # Only in the case of creating a new database we skip the check.
        if url.drivername == 'sqlite':

            #if not init and url.database is not None and not os.path.isfile(url.database):
            #    # url.database is None if opening a in-memory DB, e.g. 'sqlite://'
            #    raise Exception('SQLite database %s not found.' % url.database)

            self.engine = sqlalchemy.create_engine(url)

            enabled_foreign_keys = self.engine.execute('pragma foreign_keys').scalar()
            supports_foreign_keys = enabled_foreign_keys is not None
            if not supports_foreign_keys:
                logger.warning('Your SQLite database does not support foreign keys, so constraints will not be checked. Please upgrade.')
            elif not enabled_foreign_keys:
                self.engine.execute('pragma foreign_keys = on')

        else:
            self.engine = sqlalchemy.create_engine(url)

        self._session = sqlalchemy.orm.scoped_session(sqlalchemy.orm.sessionmaker(bind=self.engine))

        self._is_frontier = url.drivername == 'oracle+frontier'
        self._is_oracle = url.drivername == 'oracle'
        self._is_sqlite = url.drivername == 'sqlite'

        self._is_read_only = self._is_frontier or url.host in {
            'cms_orcon_adg',
            'cmsarc_lb',
        }

        self._is_official = self._is_frontier or url.host in {
            'cms_orcon_adg',
            'cmsarc_lb',
            'cms_orcoff_int',
            'cms_orcoff_prep',
            'cms_orcon_prod',
            'cmsintr_lb',
        }
        self._url = url
        self._backendName = ('sqlite' if self._is_sqlite else 'oracle' ) 
        self._schemaName = ( None if self._is_sqlite else schema_name )
        logging.debug(' ... using db "%s", schema "%s"' % (url, self._schemaName) )
        logging.debug('Loading db types...')
        self.get_dbtype(Tag).__name__
        self.get_dbtype(Payload)
        self.get_dbtype(IOV)
        self.get_dbtype(TagLog)
        self.get_dbtype(GlobalTag)
        self.get_dbtype(GlobalTagMap)

    def get_dbtype(self,theType):
        basename = theType.__name__
        if self._backendName not in db_models.keys() or basename not in db_models[self._backendName].keys():
            return make_dbtype( self._backendName, self._schemaName, theType )
        else:
            return db_models[self._backendName][basename]

    def session(self):
        s = self._session()
        s.get_dbtype = self.get_dbtype
        s._is_sqlite = self._is_sqlite
        return s

    @property
    def metadata(self):
        return _Base.metadata

    @property
    def is_frontier(self):
        return self._is_frontier

    @property
    def is_oracle(self):
        return self._is_oracle

    @property
    def is_sqlite(self):
        return self._is_sqlite

    @property
    def is_read_only(self):
        return self._is_read_only

    @property
    def is_official(self):
        return self._is_official

    def is_valid(self):
        '''Tests whether the current DB looks like a valid CMS Conditions one.
        '''
        engine_connection = self.engine.connect()
        #ret = all([self.engine.dialect.has_table(engine_connection, table.__tablename__) for table in [Tag, IOV, Payload, GlobalTag, GlobalTagMap]])
        # temporarely avoid the check on the GT tables - there are releases in use where C++ does not create these tables.
        #ret = all([self.engine.dialect.has_table(engine_connection, table.__tablename__,table.__table_args__['schema']) for table in [Tag, IOV, Payload]])
        _Tag = self.get_dbtype(Tag)
        _IOV = self.get_dbtype(IOV)
        _Payload = self.get_dbtype(Payload) 
        ret = all([self.engine.dialect.has_table(engine_connection, table.__tablename__,getSchema(table)) for table in [_Tag, _IOV, _Payload]])
        engine_connection.close()
        return ret

    def init(self, drop=False):
        '''Initializes a database.
        '''
        logging.info('Initializing database...')
        if drop:
            logging.debug('Dropping tables...')
            self.metadata.drop_all(self.engine)
        else:
            if self.is_valid():
                raise Exception('Looks like the database is already a valid CMS Conditions one.')

        logging.debug('Creating tables...')
        self.get_dbtype(Tag).__table__.create(bind = self.engine)
        self.get_dbtype(Payload).__table__.create(bind = self.engine)
        self.get_dbtype(IOV).__table__.create(bind = self.engine)
        self.get_dbtype(TagLog).__table__.create(bind = self.engine)
        self.get_dbtype(GlobalTag).__table__.create(bind = self.engine)
        self.get_dbtype(GlobalTagMap).__table__.create(bind = self.engine)
        #self.metadata.create_all(self.engine)
        if self.is_sqlite:
            # horrible hack, but no choice because of the sqlalchemy bug ( see comment in the model) 
            import sqlite3
            import string
            conn = sqlite3.connect( self._url.database )
            c = conn.cursor()
            stmt = string.Template('ALTER TABLE $before RENAME TO $after')
            c.execute( stmt.substitute( before=GlobalTag.__tablename__, after='TMP0' ) )
            c.execute( stmt.substitute( before='TMP0', after=GlobalTag.__tablename__.upper() ) )
            c.execute( stmt.substitute( before=GlobalTagMap.__tablename__, after='TMP1' ) )
            c.execute( stmt.substitute( before='TMP1', after=GlobalTagMap.__tablename__.upper() ) )
            conn.commit()
            conn.close()
        # TODO: Create indexes
        #logger.debug('Creating indexes...')


# Connection helpers
def _getCMSFrontierConnectionString(database):
    import subprocess
    return subprocess.Popen(['cmsGetFnConnect', 'frontier://%s' % database], stdout = subprocess.PIPE).communicate()[0].strip()


def _getCMSFrontierSQLAlchemyConnectionString(database, schema = 'cms_conditions'):
    import urllib
    return 'oracle+frontier://@%s/%s' % (urllib.quote_plus(_getCMSFrontierConnectionString(database)), schema)


def _getCMSOracleSQLAlchemyConnectionString(database, schema = 'cms_conditions'):
    return 'oracle://%s@%s' % (schema, database)


# Entry point

def make_url(database='pro',read_only = True):

    #schema = 'cms_conditions'  # set the default
    if ':' in database and '://' not in database: # check if we really got a shortcut like "pro:<schema>" (and not a url like proto://...), if so, disentangle
       database, schema = database.split(':')

    # Lazy in order to avoid calls to cmsGetFnConnect
    mapping = {
        'pro_R':           lambda: _getCMSFrontierSQLAlchemyConnectionString('PromptProd', schema_name),
        'arc_R':           lambda: _getCMSFrontierSQLAlchemyConnectionString('FrontierArc', schema_name),
        'int_R':           lambda: _getCMSFrontierSQLAlchemyConnectionString('FrontierInt', schema_name),
        'dev_R':           lambda: _getCMSFrontierSQLAlchemyConnectionString('FrontierPrep', schema_name),

        'orapro_R':        lambda: _getCMSOracleSQLAlchemyConnectionString('cms_orcon_adg', dbreader_user_name),
        'oraarc_R':        lambda: _getCMSOracleSQLAlchemyConnectionString('cmsarc_lb', dbreader_user_name),
        'oraint_R':        lambda: _getCMSOracleSQLAlchemyConnectionString('cms_orcoff_int', dbreader_user_name),
        'oraint_W':        lambda: _getCMSOracleSQLAlchemyConnectionString('cms_orcoff_int', dbwriter_user_name),
        'oradev_R':        lambda: _getCMSOracleSQLAlchemyConnectionString('cms_orcoff_prep', dbreader_user_name),
        'oradev_W':        lambda: _getCMSOracleSQLAlchemyConnectionString('cms_orcoff_prep', devdbwriter_user_name),

        'onlineorapro_R':  lambda: _getCMSOracleSQLAlchemyConnectionString('cms_orcon_prod', dbreader_user_name),
        'onlineorapro_W':  lambda: _getCMSOracleSQLAlchemyConnectionString('cms_orcon_prod', dbwriter_user_name),
        'onlineoraint_R':  lambda: _getCMSOracleSQLAlchemyConnectionString('cmsintr_lb', dbreader_user_name),
        'onlineoraint_W':  lambda: _getCMSOracleSQLAlchemyConnectionString('cmsintr_lb', dbwriter_user_name),
    }

    key = database + ('_R' if read_only else '_W')
    if key in mapping:
        database = mapping[key]()

    logging.debug('connection string set to "%s"' % database)

    try:
        url = sqlalchemy.engine.url.make_url(database)
    except sqlalchemy.exc.ArgumentError:
        url = sqlalchemy.engine.url.make_url('sqlite:///%s' % database)
    return url

def connect(url, init=False, authPath=None, verbose=0):
    '''Returns a Connection instance to the CMS Condition DB.

    See database_help for the description of the database parameter.

    The verbosity level is as follows:

        0 = No output (default).
        1 = SQL statements issued, including their parameters.
        2 = In addition, results of the queries (all rows and the column headers).
    '''

    if url.drivername == 'oracle' and url.password is None:
        if authPath is None:
            if authPathEnvVar in os.environ:
                authPath = os.environ[authPathEnvVar]
        authFile = None
        if authPath is not None:
            authFile = os.path.join(authPath,'.netrc')
        if authFile is not None:
            entryKey = url.host+"/"+url.username
            logging.debug('Looking up credentials for %s in file %s ' %(entryKey,authFile) )
            import netrc
            try:
                # Try to find the netrc entry
                (username, account, password) = netrc.netrc( authFile ).authenticators(entryKey)
                url.password = password
            except IOError as e:
                logging.error('.netrc file expected in %s has not been found or cannot be open.' %authPath)
                raise e
            except TypeError as e:
                logging.error('The .netrc file in %s is invalid, or the targeted entry has not been found.' %authPath)
            except Exception as e:
                logging.error('Problem with .netrc file in %s: %s' %(authPath,str(e)))     
        else:
            import getpass
            url.password = getpass.getpass('Password for %s: ' % str(url))

    if verbose >= 1:
        logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

    if verbose >= 2:
        logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)

    return Connection(url, init=init)


def _exists(session, primary_key, value):
    ret = None
    try: 
        ret = session.query(primary_key).\
    	    filter(primary_key == value).\
            count() != 0
    except sqlalchemy.exc.OperationalError:
        pass

    return ret

def _inserted_before(timestamp):
    '''To be used inside filter().
    '''

    if timestamp is None:
        # XXX: Returning None does not get optimized (skipped) by SQLAlchemy,
        #      and returning True does not work in Oracle (generates "and 1"
        #      which breaks Oracle but not SQLite). For the moment just use
        #      this dummy condition.
        return sqlalchemy.literal(True) == sqlalchemy.literal(True)

    return conddb.IOV.insertion_time <= _parse_timestamp(timestamp)

def listObject(session, name, snapshot=None):

    is_tag = _exists(session, Tag.name, name)
    result = {}
    if is_tag:
        result['type'] = 'Tag'
        result['name'] = session.query(Tag).get(name).name
	result['timeType'] = session.query(Tag.time_type).\
				     filter(Tag.name == name).\
            			     scalar()
    
        result['iovs'] = session.query(IOV.since, IOV.insertion_time, IOV.payload_hash, Payload.object_type).\
                join(IOV.payload).\
                filter(
                    IOV.tag_name == name,
                    _inserted_before(snapshot),
                ).\
                order_by(IOV.since.desc(), IOV.insertion_time.desc()).\
                from_self().\
                order_by(IOV.since, IOV.insertion_time).\
                all()

    try:
        is_global_tag = _exists(session, GlobalTag.name, name)
        if is_global_tag:
            result['type'] = 'GlobalTag'
	    result['name'] = session.query(GlobalTag).get(name)
            result['tags'] = session.query(GlobalTagMap.record, GlobalTagMap.label, GlobalTagMap.tag_name).\
                                     filter(GlobalTagMap.global_tag_name == name).\
                    		     order_by(GlobalTagMap.record, GlobalTagMap.label).\
                    		     all()
    except sqlalchemy.exc.OperationalError:
        sys.stderr.write("No table for GlobalTags found in DB.\n\n")

    if not is_tag and not is_global_tag:
        raise Exception('There is no tag or global tag named %s in the database.' % name)

    return result

def getPayload(session, hash):
    # get payload from DB:
    data, payloadType = session.query(Payload.data, Payload.object_type).filter(Payload.hash == hash).one()
    return data
