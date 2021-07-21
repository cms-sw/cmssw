'''CMS Conditions DB Python library.
'''

__author__ = 'Miguel Ojeda'
__copyright__ = 'Copyright 2013, CERN'
__credits__ = ['Giacomo Govi', 'Miguel Ojeda', 'Andreas Pfeiffer']
__license__ = 'Unknown'
__maintainer__ = 'Giacomo Govi'
__email__ = 'giacomo.govi@cern.ch'


import os
import hashlib
import logging

import sqlalchemy
import sqlalchemy.ext.declarative
import enum
from sqlalchemy import Enum

schema_name = 'cms_conditions'
dbuser_name = 'cms_conditions'
dbreader_user_name = 'cms_cond_general_r'
dbwriter_user_name = 'cms_cond_general_w'
logger = logging.getLogger(__name__)

#authentication/authorization params
authPathEnvVar = 'COND_AUTH_PATH'
dbkey_filename = 'db.key'
dbkey_folder = os.path.join('.cms_cond',dbkey_filename)

# frontier services
PRO ='PromptProd'
ARC ='FrontierArc'
INT ='FrontierInt'
DEV ='FrontierPrep'
# oracle read only services
ORAPRO = 'cms_orcon_adg'
ORAARC = 'cmsarc_lb'
# oracle masters
ORAINT = 'cms_orcoff_int'
ORADEV = 'cms_orcoff_prep'
ONLINEORAPRO = 'cms_orcon_prod'
ONLINEORAINT = 'cmsintr_lb'

# Set initial level to WARN.  This so that log statements don't occur in
# the absense of explicit logging being enabled.
if logger.level == logging.NOTSET:
    logger.setLevel(logging.WARN)

# Utility functions
def hash(data):
    return hashlib.sha1(data.encode('ascii')).hexdigest()


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

def oracle_connection_string(db_service, db_schema ):
    return 'oracle://%s/%s'%(db_service,db_schema)

class Synchronization(enum.Enum):
    any        = 'any'
    validation = 'validation'
    mc         = 'mc'
    runmc      = 'runmc'
    hlt        = 'hlt'
    express    = 'express'
    prompt     = 'prompt'
    pcl        = 'pcl'
    offline    = 'offline'

synch_list = list(x.value for x in list(Synchronization))

class TimeType(enum.Enum):
    Run  = 'Run'
    Time = 'Time'
    Lumi = 'Lumi'
    Hash = 'Hash'
    User = 'User'


# Schema definition
_Base = sqlalchemy.ext.declarative.declarative_base()

def fq_name( schema_name, table_name ):
    name = table_name
    if schema_name is not None:
        name = '%s.%s' %(schema_name, table_name)
    return name

db_models = {}

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
        defColVal = None
        if len(v)==3:
            defColVal = v[2]
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
                if defColVal is None:
                    members[k] = sqlalchemy.Column(v[0],nullable=nullable)
                else:
                    members[k] = sqlalchemy.Column(v[0],nullable=nullable, default=defColVal)
    dbType = type(dbtype_name,(_Base,),members)

    if backendName not in db_models.keys():
        db_models[backendName] = {}
    db_models[backendName][baseType.__name__] = dbType
    return dbType

def getSchema(tp):
    if tp.__table_args__ is not None:
        return tp.__table_args__['schema']
    return None

class Tag:
    __tablename__       = 'TAG'
    columns             = { 'name': (sqlalchemy.String(name_length),_Col.pk), 
                            'time_type': (sqlalchemy.Enum(*tuple(TimeType.__members__.keys())),_Col.notNull),
                            'object_type': (sqlalchemy.String(name_length),_Col.notNull),
                            'synchronization': (sqlalchemy.Enum(*tuple(Synchronization.__members__.keys())),_Col.notNull),
                            'description': (sqlalchemy.String(description_length),_Col.notNull),
                            'last_validated_time':(sqlalchemy.BIGINT,_Col.notNull),
                            'end_of_validity':(sqlalchemy.BIGINT,_Col.notNull),
                            'insertion_time':(sqlalchemy.TIMESTAMP,_Col.notNull),
                            'modification_time':(sqlalchemy.TIMESTAMP,_Col.notNull),
                            'protection_code':(sqlalchemy.Integer,_Col.notNull,0)   }

class TagMetadata:
    __tablename__       = 'TAG_METADATA'
    columns             = { 'tag_name': (DbRef(Tag,'name'),_Col.pk), 
                            'min_serialization_v': (sqlalchemy.String(20),_Col.notNull),
                            'min_since': (sqlalchemy.BIGINT,_Col.notNull),
                            'modification_time':(sqlalchemy.TIMESTAMP,_Col.notNull) }

class TagAuthorization:
    __tablename__       = 'TAG_AUTHORIZATION'
    columns             = { 'tag_name': (DbRef(Tag,'name'),_Col.pk), 
                            'access_type': (sqlalchemy.Integer,_Col.notNull),
                            'credential': (sqlalchemy.String(name_length),_Col.notNull),
                            'credential_type':(sqlalchemy.Integer,_Col.notNull) }

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
                            'payload_hash':(DbRef(Payload,'hash'),_Col.notNull) }


class GlobalTag:
    __tablename__       = 'GLOBAL_TAG'
    columns             = { 'name':(sqlalchemy.String(name_length),_Col.pk),
                            'validity': (sqlalchemy.BIGINT,_Col.notNull),
                            'description':(sqlalchemy.String(description_length),_Col.notNull),
                            'release':(sqlalchemy.String(name_length),_Col.notNull),
                            'insertion_time':(sqlalchemy.TIMESTAMP,_Col.notNull),
                            'snapshot_time':(sqlalchemy.TIMESTAMP,_Col.notNull) }

class GlobalTagMap:
    __tablename__       = 'GLOBAL_TAG_MAP'
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

class RunInfo:
    __tablename__       = 'RUN_INFO'
    columns             = { 'run_number':(sqlalchemy.BIGINT,_Col.pk),
                            'start_time':(sqlalchemy.TIMESTAMP,_Col.notNull),
                            'end_time':(sqlalchemy.TIMESTAMP,_Col.notNull) }

class BoostRunMap:
    __tablename__       = 'BOOST_RUN_MAP'
    columns             = { 'run_number':(sqlalchemy.BIGINT,_Col.pk),
                            'run_start_time':(sqlalchemy.TIMESTAMP,_Col.notNull),
                            'boost_version': (sqlalchemy.String(20),_Col.notNull) }

# CondDB object
class Connection(object):

    def __init__(self, url):
        # Workaround to avoid creating files if not present.
        # Python's sqlite3 module does not use sqlite3_open_v2(),
        # and therefore we cannot disable SQLITE_OPEN_CREATE.
        # Only in the case of creating a new database we skip the check.
        if url.drivername == 'sqlite':

            self.engine = sqlalchemy.create_engine(url)

            enabled_foreign_keys = self.engine.execute('pragma foreign_keys').scalar()
            supports_foreign_keys = enabled_foreign_keys is not None
            if not supports_foreign_keys:
                logger.warning('Your SQLite database does not support foreign keys, so constraints will not be checked. Please upgrade.')
            elif not enabled_foreign_keys:
                self.engine.execute('pragma foreign_keys = on')

        else:
            self.engine = sqlalchemy.create_engine(url, max_identifier_length=30)

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
        logging.debug('Loading db types...')
        self.get_dbtype(Tag).__name__
        self.get_dbtype(Payload)
        self.get_dbtype(IOV)
        self.get_dbtype(TagLog)
        self.get_dbtype(GlobalTag)
        self.get_dbtype(GlobalTagMap)
        self.get_dbtype(RunInfo)
        if not self._is_sqlite:
            self.get_dbtype(TagMetadata)
            self.get_dbtype(TagAuthorization)
            self.get_dbtype(BoostRunMap)
        self._is_valid = self.is_valid()

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
        s.is_oracle = self.is_oracle
        s._url = self._url
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
        # temporarely avoid the check on the GT tables - there are releases in use where C++ does not create these tables.
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
            self._is_valid = False
        else:
            if not self._is_valid:
                logging.debug('Creating tables...')
                self.get_dbtype(Tag).__table__.create(bind = self.engine)
                self.get_dbtype(Payload).__table__.create(bind = self.engine)
                self.get_dbtype(IOV).__table__.create(bind = self.engine)
                self.get_dbtype(TagLog).__table__.create(bind = self.engine)
                self.get_dbtype(GlobalTag).__table__.create(bind = self.engine)
                self.get_dbtype(GlobalTagMap).__table__.create(bind = self.engine)
                self._is_valid = True

def getSessionOnMasterDB( session1, session2 ):
    key = '%s/%s' 
    sessiondict = { }
    sessiondict[key %(session1._url.drivername,session1._url.host)] = session1
    sessiondict[key %(session2._url.drivername,session2._url.host)] = session2
    masterkey = key %('oracle',ONLINEORAPRO)
    if masterkey in sessiondict.keys():
        return sessiondict[masterkey]
    adgkey = key %('oracle',ORAPRO)
    if adgkey in sessiondict.keys():
        return sessiondict[adgkey]
    frontierkey = key %('frontier',PRO)
    if frontierkey in sessiondict.keys():
        return sessiondict[frontierkey]
    # default case: frontier on pro
    conn = Connection(make_url())
    session = conn.session()
    # is it required?
    session._conn = conn
    return session

# Connection helpers
def _getCMSFrontierConnectionString(database):
    import subprocess
    return subprocess.Popen(['cmsGetFnConnect', 'frontier://%s' % database], stdout = subprocess.PIPE).communicate()[0].strip()

def _getCMSSQLAlchemyConnectionString(technology,service,schema_name):
    if technology == 'frontier':
        import urllib
        import sys
        py3k = sys.version_info >= (3, 0)        
        if py3k:
            return '%s://@%s/%s' % ('oracle+frontier', urllib.parse.quote_plus(_getCMSFrontierConnectionString(service)), schema_name )
        else:
            return '%s://@%s/%s' % ('oracle+frontier', urllib.quote_plus(_getCMSFrontierConnectionString(service)), schema_name )
    elif technology == 'oracle':
        return '%s://%s@%s' % (technology, schema_name, service)

# Entry point
def make_url(database='pro',read_only = True):
    if database.startswith('sqlite:') or database.startswith('sqlite_file:'):
        ignore, database = database.split(':',1)

    if ':' in database and '://' not in database: # check if we really got a shortcut like "pro:<schema>" (and not a url like proto://...), if so, disentangle
        database, schema = database.split(':')

    officialdbs = { 
        # frontier 
        'pro' :         ('frontier','PromptProd',             { 'R': schema_name }, ),
        'arc' :         ('frontier','FrontierArc',            { 'R': schema_name }, ),
        'int' :         ('frontier','FrontierInt',            { 'R': schema_name }, ),
        'dev' :         ('frontier','FrontierPrep',           { 'R': schema_name }, ),
        # oracle adg
        'orapro':       ('oracle',         'cms_orcon_adg',   { 'R': dbreader_user_name }, ),
        'oraarc':       ('oracle',         'cmsarc_lb',       { 'R': dbreader_user_name }, ),
        # oracle masters
        'oraint':       ('oracle',         'cms_orcoff_int',  { 'R': dbreader_user_name,
                                                                'W': dbwriter_user_name }, ),
        'oradev':       ('oracle',         'cms_orcoff_prep', { 'R': dbreader_user_name,
                                                                'W': dbwriter_user_name }, ),
        'onlineorapro': ('oracle',         'cms_orcon_prod',  { 'R': dbreader_user_name,
                                                                'W': dbwriter_user_name }, ),
        'onlineoraint': ('oracle',         'cmsintr_lb',      { 'R': dbreader_user_name,
                                                                'W': dbwriter_user_name }, ),
    }

    if database in officialdbs.keys():
        key = ('R' if read_only else 'W')
        mapping = officialdbs[database]
        tech = mapping[0]
        service = mapping[1]
        schema_dict = mapping[2]
        if key in schema_dict.keys():
            database = _getCMSSQLAlchemyConnectionString(tech,service,schema_dict[key])
        else:
            raise Exception("Read-only database %s://%s cannot be accessed in update mode." %(tech,service))

    logging.debug('connection string set to "%s"' % database)

    try:
        url = sqlalchemy.engine.url.make_url(database)
    except sqlalchemy.exc.ArgumentError:
        url = sqlalchemy.engine.url.make_url('sqlite:///%s' % database)
    return url

def connect(url, authPath=None, verbose=0, as_admin=False):
    '''Returns a Connection instance to the CMS Condition DB.

    See database_help for the description of the database parameter.

    The verbosity level is as follows:

        0 = No output (default).
        1 = SQL statements issued, including their parameters.
        2 = In addition, results of the queries (all rows and the column headers).
    '''

    check_admin = as_admin
    if url.drivername == 'oracle':
        if url.username is None:
            logging.error('Could not resolve the username for the connection %s. Please provide a connection in the format oracle://[user]:[pass]@[host]' %url )
            raise Exception('Connection format error: %s' %url )
        if url.password is None:
            if authPath is None:
                if authPathEnvVar in os.environ:
                    authPath = os.environ[authPathEnvVar]
            explicit_auth = False
            if authPath is not None:
                dbkey_path = os.path.join(authPath,dbkey_folder)
                if not os.path.exists(dbkey_path):
                    authFile = os.path.join(authPath,'.netrc')
                    if os.path.exists(authFile):
                        entryKey = url.host.lower()+"/"+url.username.lower()
                        logging.debug('Looking up credentials for %s in file %s ' %(entryKey,authFile) )
                        import netrc
                        params = netrc.netrc( authFile ).authenticators(entryKey)
                        if params is not None:
                            (username, account, password) = params
                            url.username = username
                            url.password = password
                        else:
                            msg = 'The entry %s has not been found in the .netrc file.' %entryKey
                            raise TypeError(msg)
                    else:
                        explicit_auth =True
                else:
                    import libCondDBPyBind11Interface as auth
                    role_code = auth.reader_role
                    if url.username == dbwriter_user_name:
                        role_code = auth.writer_role
                    if check_admin:
                        role_code = auth.admin_role
                    connection_string = oracle_connection_string(url.host.lower(),schema_name)
                    logging.debug('Using db key to get credentials for %s' %connection_string )
                    (dbuser,username,password) = auth.get_credentials_from_db(connection_string,role_code,authPath)
                    if username=='' or password=='':
                        raise Exception('No credentials found to connect on %s with the required access role.'%connection_string)
                    check_admin = False
                    url.username = username
                    url.password = password
            else:
                import getpass
                pwd = getpass.getpass('Password for %s: ' % str(url))
                if pwd is None or pwd == '':
                    pwd = getpass.getpass('Password for %s: ' % str(url))
                    if pwd is None or pwd == '':
                        raise Exception('Empty password provided, bailing out...')
                url.password = pwd
        if check_admin:
            raise Exception('Admin access has not been granted. Please provide a valid admin db-key.')
    if check_admin:
       raise Exception('Admin access is not available for technology "%s".' %url.drivername)
    if verbose >= 1:
        logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

    if verbose >= 2:
        logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)

    return Connection(url)


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
