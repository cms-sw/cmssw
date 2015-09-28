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
      oraboost      Production   Oracle (ADG)   read-write  Password required.
      oraboostprep  Development  Oracle         read-write  Password required.

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
    offline = 'Offline'
    hlt     = 'HLT'
    prompt  = 'Prompt'


class TimeType(Enum):
    run  = 'Run'
    time = 'Time'
    lumi = 'Lumi'
    hash = 'Hash'
    user = 'User'


# Schema definition
_Base = sqlalchemy.ext.declarative.declarative_base()


class Tag(_Base):
    __tablename__       = 'TAG'

    name                = sqlalchemy.Column(sqlalchemy.String(name_length),           primary_key=True)
    time_type           = sqlalchemy.Column(sqlalchemy.Enum(*tuple(TimeType)),        nullable=False)
    object_type         = sqlalchemy.Column(sqlalchemy.String(name_length),           nullable=False)
    synchronization     = sqlalchemy.Column(sqlalchemy.Enum(*tuple(Synchronization)), nullable=False)
    description         = sqlalchemy.Column(sqlalchemy.String(description_length),    nullable=False)
    last_validated_time = sqlalchemy.Column(sqlalchemy.BIGINT,                       nullable=False)
    end_of_validity     = sqlalchemy.Column(sqlalchemy.BIGINT,                       nullable=False)
    insertion_time      = sqlalchemy.Column(sqlalchemy.TIMESTAMP,                     nullable=False)
    modification_time   = sqlalchemy.Column(sqlalchemy.TIMESTAMP,                     nullable=False)

    iovs                = sqlalchemy.orm.relationship('IOV')


class IOV(_Base):
    __tablename__       = 'IOV'

    tag_name            = sqlalchemy.Column(sqlalchemy.ForeignKey('TAG.name'),        primary_key=True)
    since               = sqlalchemy.Column(sqlalchemy.BIGINT,                       primary_key=True)
    insertion_time      = sqlalchemy.Column(sqlalchemy.TIMESTAMP,                     primary_key=True)
    payload_hash        = sqlalchemy.Column(sqlalchemy.ForeignKey('PAYLOAD.hash'),    nullable=False)

    tag                 = sqlalchemy.orm.relationship('Tag')
    payload             = sqlalchemy.orm.relationship('Payload')


class Payload(_Base):
    __tablename__       = 'PAYLOAD'

    hash                = sqlalchemy.Column(sqlalchemy.CHAR(hash_length),             primary_key=True)
    object_type         = sqlalchemy.Column(sqlalchemy.String(name_length),           nullable=False)
    data                = sqlalchemy.Column(sqlalchemy.BLOB,                          nullable=False)
    streamer_info       = sqlalchemy.Column(sqlalchemy.BLOB,                          nullable=False)
    version             = sqlalchemy.Column(sqlalchemy.String(20),                    nullable=False)
    insertion_time      = sqlalchemy.Column(sqlalchemy.TIMESTAMP,                     nullable=False)


class GlobalTag(_Base):
    __tablename__       = 'GLOBAL_TAG'

    name                = sqlalchemy.Column(sqlalchemy.String(name_length),           primary_key=True)
    validity            = sqlalchemy.Column(sqlalchemy.BIGINT,                       nullable=False)
    description         = sqlalchemy.Column(sqlalchemy.String(description_length),    nullable=False)
    release             = sqlalchemy.Column(sqlalchemy.String(name_length),           nullable=False)
    insertion_time      = sqlalchemy.Column(sqlalchemy.TIMESTAMP,                     nullable=False)
    snapshot_time       = sqlalchemy.Column(sqlalchemy.TIMESTAMP,                     nullable=False)


class GlobalTagMap(_Base):
    __tablename__       = 'GLOBAL_TAG_MAP'

    global_tag_name     = sqlalchemy.Column(sqlalchemy.ForeignKey('GLOBAL_TAG.name'), primary_key=True)
    record              = sqlalchemy.Column(sqlalchemy.String(name_length),           primary_key=True)
    label               = sqlalchemy.Column(sqlalchemy.String(name_length),           primary_key=True)
    tag_name            = sqlalchemy.Column(sqlalchemy.ForeignKey('TAG.name'),        nullable=False)

    global_tag          = sqlalchemy.orm.relationship('GlobalTag')
    tag                 = sqlalchemy.orm.relationship('Tag')


# CondDB object
class Connection(object):

    def __init__(self, url, init=False):
        # Workaround to avoid creating files if not present.
        # Python's sqlite3 module does not use sqlite3_open_v2(),
        # and therefore we cannot disable SQLITE_OPEN_CREATE.
        # Only in the case of creating a new database we skip the check.
        if url.drivername == 'sqlite':

            if not init and url.database is not None and not os.path.isfile(url.database):
                # url.database is None if opening a in-memory DB, e.g. 'sqlite://'
                raise Exception('SQLite database %s not found.' % url.database)

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

    def session(self):
        return self._session()

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
        ret = all([self.engine.dialect.has_table(engine_connection, table.__tablename__) for table in [Tag, IOV, Payload, GlobalTag, GlobalTagMap]])
        engine_connection.close()
        return ret

    def init(self, drop=False):
        '''Initializes a database.
        '''

        if drop:
            logger.debug('Dropping tables...')
            self.metadata.drop_all(self.engine)
        else:
            if self.is_valid():
                raise Exception('Looks like the database is already a valid CMS Conditions one. Please use drop=True if you really want to scratch it.')

        logger.debug('Creating tables...')
        self.metadata.create_all(self.engine)

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

def make_url(database='pro'):

    schema = 'cms_conditions'  # set the default
    if ':' in database and '://' not in database: # check if we really got a shortcut like "pro:<schema>" (and not a url like proto://...), if so, disentangle
       database, schema = database.split(':')
    logging.debug(' ... using db "%s", schema "%s"' % (database, schema) )

    # Lazy in order to avoid calls to cmsGetFnConnect
    mapping = {
        'pro':           lambda: _getCMSFrontierSQLAlchemyConnectionString('PromptProd', schema),
        'arc':           lambda: _getCMSFrontierSQLAlchemyConnectionString('FrontierArc', schema),
        'int':           lambda: _getCMSFrontierSQLAlchemyConnectionString('FrontierInt', schema),
        'dev':           lambda: _getCMSFrontierSQLAlchemyConnectionString('FrontierPrep', schema),

        'orapro':        lambda: _getCMSOracleSQLAlchemyConnectionString('cms_orcon_adg', schema),
        'oraarc':        lambda: _getCMSOracleSQLAlchemyConnectionString('cmsarc_lb', schema),
        'oraint':        lambda: _getCMSOracleSQLAlchemyConnectionString('cms_orcoff_int', schema),
        'oradev':        lambda: _getCMSOracleSQLAlchemyConnectionString('cms_orcoff_prep', schema),

        'onlineorapro':  lambda: _getCMSOracleSQLAlchemyConnectionString('cms_orcon_prod', schema),
        'onlineoraint':  lambda: _getCMSOracleSQLAlchemyConnectionString('cmsintr_lb', schema),
    }

    if database in mapping:
        database = mapping[database]()

    logging.debug('connection string set to "%s"' % database)

    try:
        url = sqlalchemy.engine.url.make_url(database)
    except sqlalchemy.exc.ArgumentError:
        url = sqlalchemy.engine.url.make_url('sqlite:///%s' % database)
    return url

def connect(url, init=False, verbose=0):
    '''Returns a Connection instance to the CMS Condition DB.

    See database_help for the description of the database parameter.

    The verbosity level is as follows:

        0 = No output (default).
        1 = SQL statements issued, including their parameters.
        2 = In addition, results of the queries (all rows and the column headers).
    '''

    if url.drivername == 'oracle' and url.password is None:
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
