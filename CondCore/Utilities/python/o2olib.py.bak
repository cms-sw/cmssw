__author__ = 'Giacomo Govi'

import sqlalchemy
import sqlalchemy.ext.declarative
import subprocess
from datetime import datetime
import os
import sys
import logging
import string
import json

import CondCore.Utilities.credentials as auth

prod_db_service = ['cms_orcon_prod','cms_orcon_prod/cms_cond_general_w']
dev_db_service = ['cms_orcoff_prep','cms_orcoff_prep/cms_test_conditions']
schema_name = 'CMS_CONDITIONS'
sqlalchemy_tpl = 'oracle://%s:%s@%s'
coral_tpl = 'oracle://%s/%s'
private_db = 'sqlite:///o2o_jobs.db'
startStatus = -1
authPathEnvVar = 'COND_AUTH_PATH'
messageLevelEnvVar = 'O2O_LOG_LEVEL'
logFolderEnvVar = 'O2O_LOG_FOLDER'

_Base = sqlalchemy.ext.declarative.declarative_base()

fmt_str = "[%(asctime)s] %(levelname)s: %(message)s"
logLevel = logging.INFO
if messageLevelEnvVar in os.environ:
    levStr = os.environ[messageLevelEnvVar]
    if levStr == 'DEBUG':
        logLevel = logging.DEBUG
logFormatter = logging.Formatter(fmt_str)

class O2OJob(_Base):
    __tablename__      = 'O2O_JOB'
    __table_args__     = {'schema' : schema_name}
    name               = sqlalchemy.Column(sqlalchemy.String(100),    primary_key=True)
    enabled            = sqlalchemy.Column(sqlalchemy.Integer,        nullable=False)
    tag_name           = sqlalchemy.Column(sqlalchemy.String(100),    nullable=False)
    interval           = sqlalchemy.Column(sqlalchemy.Integer,        nullable=False)

class O2OJobConf(_Base):
    __tablename__      = 'O2O_JOB_CONF'
    __table_args__     = {'schema' : schema_name}
    job_name           = sqlalchemy.Column(sqlalchemy.ForeignKey(O2OJob.name),    primary_key=True)
    insertion_time     = sqlalchemy.Column(sqlalchemy.TIMESTAMP,      primary_key=True)
    configuration      = sqlalchemy.Column(sqlalchemy.String(4000),   nullable=False)

    job                = sqlalchemy.orm.relationship('O2OJob', primaryjoin="O2OJob.name==O2OJobConf.job_name")

class O2ORun(_Base):
    __tablename__      = 'O2O_RUN'
    __table_args__     = {'schema' : schema_name}
    job_name           = sqlalchemy.Column(sqlalchemy.ForeignKey(O2OJob.name),    primary_key=True)
    start_time         = sqlalchemy.Column(sqlalchemy.TIMESTAMP,      primary_key=True)
    end_time           = sqlalchemy.Column(sqlalchemy.TIMESTAMP,      nullable=True)
    status_code        = sqlalchemy.Column(sqlalchemy.Integer,        nullable=False)
    log                = sqlalchemy.Column(sqlalchemy.CLOB,           nullable=True)

    job                = sqlalchemy.orm.relationship('O2OJob', primaryjoin="O2OJob.name==O2ORun.job_name")

def get_db_credentials( db_service, authFile ):
    (username, account, pwd) = auth.get_credentials( authPathEnvVar, db_service[1], authFile )
    return username,pwd

def print_table( headers, table ):
    ws = []
    for h in headers:
        ws.append(len(h))
    for row in table:
        ind = 0
        for c in row:
            if ind<len(ws):
                if len(c)> ws[ind]:
                    ws[ind] = len(c)
            ind += 1

    def printf( row ):
        line = ''
        ind = 0
        for w in ws:
            fmt = '{:<%s}' %w
            if ind<len(ws):
                line += (fmt.format( row[ind] )+' ') 
            ind += 1
        print line
    printf( headers )
    hsep = ''
    for w in ws:
        fmt = '{:-<%s}' %w
        hsep += (fmt.format('')+' ')
    print hsep
    for row in table:
        printf( row )

class O2OMgr(object):
    def __init__(self):
        self.logger = logging.getLogger()        
        self.logger.setLevel(logLevel)
        consoleHandler = logging.StreamHandler(sys.stdout) 
        consoleHandler.setFormatter(logFormatter)
        self.logger.addHandler(consoleHandler)
        self.eng = None

    def logger( self ):
        return self.logger
        
    def getSession( self, db_service, auth ):
        url = None
        if db_service is None:
            url = private_db
        else:
            self.logger.info('Getting credentials')
            if auth is not None:
                if not os.path.exists(auth):
                    self.logger.error('Authentication path %s is invalid.' %auth)
                    return None
            try:
                username, pwd = get_db_credentials( db_service, auth )
            except Exception as e:
                logging.debug(str(e))
                username = None
                pwd = None
            if username is None:
                logging.error('Credentials for service %s (machine=%s) are not available' %(db_service[0],db_service[1]))
                raise Exception("Cannot connect to db %s" %db_service[0] )
            url = sqlalchemy_tpl %(username,pwd,db_service[0])
        session = None
        try:
            self.eng = sqlalchemy.create_engine( url )
            session = sqlalchemy.orm.scoped_session( sqlalchemy.orm.sessionmaker(bind=self.eng))
        except sqlalchemy.exc.SQLAlchemyError as dberror:
            self.logger.error( str(dberror) )
        return session

class O2OJobMgr(O2OMgr):

    def __init__( self ):
        O2OMgr.__init__(self)
        self.db_connection = None
        self.conf_dict = {}


    def readConfiguration( self, config_filename ):
        config = ''
        try:
            with open( config_filename, 'r' ) as config_file:
                config = config_file.read()
                if config == '':
                    O2OMgr.logger( self).error( 'The file %s contains an empty string.', config_filename )
                else:
                    json.loads(config)
        except IOError as e:
            O2OMgr.logger( self).error( 'The file %s cannot be open.', config_filename )
        except ValueError as e:
            config = ''
            O2OMgr.logger( self).error( 'The file %s contains an invalid json string.', config_filename )
        return config

    def connect( self, service, args ):
        self.session = O2OMgr.getSession( self,service, args.auth )
        self.verbose = args.verbose
        if self.session is None:
            return False
        else:
            self.db_connection = coral_tpl %(service[0],schema_name)
            self.conf_dict['db']=self.db_connection
            return True
    def runManager( self ):
        return O2ORunMgr( self.db_connection, self.session, O2OMgr.logger( self ) )

    def add( self, job_name, config_filename, int_val, en_flag ):
        res = self.session.query(O2OJob.enabled).filter_by(name=job_name)
        enabled = None
        for r in res:
            enabled = r
        if enabled:
            O2OMgr.logger( self).error( "A job called '%s' exists already.", job_name )
            return False
        configJson = self.readConfiguration( config_filename )
        if configJson == '':
            return False       
        job = O2OJob(name=job_name,tag_name='-',enabled=en_flag,interval=int_val)
        config = O2OJobConf( job_name=job_name, insertion_time = datetime.now(), configuration = configJson ) 
        self.session.add(job)
        self.session.add(config)
        self.session.commit()
        O2OMgr.logger( self).info( "New o2o job '%s' created.", job_name )
        return True 

    def set( self, job_name, en_flag ):
        res = self.session.query(O2OJob.enabled).filter_by(name=job_name)
        enabled = None
        for r in res:
            enabled = r
        if enabled is None:
            O2OMgr.logger( self).error( "A job called '%s' does not exist.", job_name )
            return
        job = O2OJob(name=job_name,enabled=en_flag)
        action = 'enabled'
        if not en_flag:
            action = 'disabled'
        self.session.merge(job)
        self.session.commit()
        O2OMgr.logger( self).info( "Job '%s' %s." %(job_name,action) )

    def setConfig( self, job_name, config_filename ):
        res = self.session.query(O2OJob.enabled).filter_by(name=job_name)
        enabled = None
        for r in res:
            enabled = r
        if enabled is None:
            O2OMgr.logger( self).error( "A job called '%s' does not exist.", job_name )
            return
        configJson = self.readConfiguration( config_filename )
        if configJson == '':
            return False       
        config = O2OJobConf( job_name=job_name, insertion_time = datetime.now(), configuration = configJson ) 
        self.session.add(config)
        self.session.commit()
        O2OMgr.logger( self).info( "New configuration inserted for job '%s'", job_name )

    def setInterval( self, job_name, int_val ):
        res = self.session.query(O2OJob.enabled).filter_by(name=job_name)
        enabled = None
        for r in res:
            enabled = r
        if enabled is None:
            O2OMgr.logger( self).error( "A job called '%s' does not exist.", job_name )
            return
        job = O2OJob(name=job_name,interval=int_val)
        self.session.merge(job)
        self.session.commit()
        O2OMgr.logger( self).info( "The execution interval for job '%s' has been updated.", job_name )

    def migrateConfig( self ):
        res = self.session.query(O2OJob.name,O2OJob.tag_name)
        for r in res:
            job_name = r[0]
            tag = r[1] 
            if tag != '-':
                configDict = {}
                configDict['tag']=tag
                configJson = json.dumps( configDict )
                config = O2OJobConf( job_name=job_name, insertion_time = datetime.now(), configuration = configJson ) 
                self.session.add(config)
                O2OMgr.logger( self).info( "Configuration for job '%s' inserted.", job_name )
        self.session.commit()
 
    def listJobs( self ):
        runs = {}
        res = self.session.query(O2ORun.job_name,sqlalchemy.func.max(O2ORun.start_time)).group_by(O2ORun.job_name).order_by(O2ORun.job_name)
        for r in res:
            runs[r[0]] = str(r[1])
        res = self.session.query(O2OJob.name, O2OJob.interval, O2OJob.enabled).order_by(O2OJob.name).all()
        table = []
        for r in res:
            row = []
            row.append(r[0]),
            row.append('%5d' %r[1] )
            enabled = 'Y' if (r[2]==1) else 'N'
            row.append('%4s' %enabled )
            lastRun = '-'
            if r[0] in runs.keys():
                lastRun = runs[r[0]]
            row.append( lastRun )
            table.append(row)
        headers = ['Job name','Interval','Enabled','Last run start']
        print_table( headers, table ) 

    def listConfig( self, jname ):
        res = self.session.query(O2OJob.enabled).filter_by(name=jname)
        enabled = None
        for r in res:
            enabled = r
        if enabled is None:
            O2OMgr.logger( self).error( "A job called '%s' does not exist.", jname )
            return
        res = self.session.query( O2OJobConf.configuration, O2OJobConf.insertion_time  ).filter_by(job_name=jname).order_by(O2OJobConf.insertion_time)
        configs = []
        for r in res:
            configs.append((str(r[0]),r[1]))
        ind = len(configs)
        if ind:
            print "Configurations for job '%s'" %jname
            for cf in reversed(configs):
                print '#%2d  since: %s' %(ind,cf[1])
                print cf[0]
                ind -= 1
        else:
            O2OMgr.logger( self ).info("No configuration found for job '%s'" %jname )

    def dumpConfig( self, jname, versionIndex, configFile ):
        versionIndex = int(versionIndex)
        res = self.session.query(O2OJob.enabled).filter_by(name=jname)
        enabled = None
        for r in res:
            enabled = r
        if enabled is None:
            O2OMgr.logger( self).error( "A job called '%s' does not exist.", jname )
            return
        res = self.session.query( O2OJobConf.configuration, O2OJobConf.insertion_time  ).filter_by(job_name=jname).order_by(O2OJobConf.insertion_time)
        configs = []
        for r in res:
            configs.append((str(r[0]),r[1]))
        ind = len(configs)
        if versionIndex>ind or versionIndex==0:
            O2OMgr.logger( self ).error("Configuration for job %s with index %s has not been found." %(jname,versionIndex))
            return
        print "Configuration #%2d for job '%s'" %(versionIndex,jname)
        config = configs[versionIndex-1]
        print '#%2d  since %s' %(versionIndex,config[1])
        print config[0]
        if configFile is None or configFile == '':
            configFile = '%s_%s.json' %(jname,versionIndex)
        with open(configFile,'w') as json_file:
            json_file.write(config[0])

            
class O2ORunMgr(object):

    def __init__( self, db_connection, session, logger ):
        self.job_name = None
        self.start = None
        self.end = None
        self.conf_dict = {}
        self.conf_dict['db'] = db_connection
        self.session = session
        self.logger = logger

    def startJob( self, job_name ):
        self.logger.info('Checking job %s', job_name)
        exists = None
        enabled = None
        try:
            res = self.session.query(O2OJob.enabled,O2OJob.tag_name).filter_by(name=job_name)
            for r in res:
                exists = True
                enabled = int(r[0])
                self.tag_name = str(r[1]) 
            if exists is None:
                self.logger.error( 'The job %s is unknown.', job_name )
                return 2
            if enabled:
                res = self.session.query(O2OJobConf.configuration).filter_by(job_name=job_name).order_by(sqlalchemy.desc(O2OJobConf.insertion_time)).first()
                conf = None
                for r in res:
                    conf = str(r)
                if conf is None:
                    self.logger.warning("No configuration found for job '%s'" %job_name )
                else:
                    try:
                        self.conf_dict.update( json.loads(conf) )
                        self.logger.info('Using configuration: %s ' %conf)
                    except Exception as e:
                        self.logger.error( str(e) )
                        return 6
                self.job_name = job_name
                self.start = datetime.now()
                run = O2ORun(job_name=self.job_name,start_time=self.start,status_code=startStatus)
                self.session.add(run)
                self.session.commit()
                return 0
            else:
                self.logger.info( 'The job %s has been disabled.', job_name )
                return 5
        except sqlalchemy.exc.SQLAlchemyError as dberror:
                self.logger.error( str(dberror) )
                return 7
        return -1


    def endJob( self, status, log ):
        self.end = datetime.now()
        try:
            run = O2ORun(job_name=self.job_name,start_time=self.start,end_time=self.end,status_code=status,log=log)
            self.session.merge(run)
            self.session.commit()
            self.logger.info( 'Job %s ended.', self.job_name )
            return 0
        except sqlalchemy.exc.SQLAlchemyError as dberror:
            self.logger.error( str(dberror) )
            return 8

    def executeJob( self, args ):
        job_name = args.name
        command = args.executable
        logFolder = os.getcwd()
        if logFolderEnvVar in os.environ:
            logFolder = os.environ[logFolderEnvVar]
        datelabel = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        logFileName = '%s-%s.log' %(job_name,datelabel)
        logFile = os.path.join(logFolder,logFileName)
        started = self.startJob( job_name )
        if started !=0:
            return started
        ret = -1
        try:
            # replacing %([key])s placeholders... 
            command = command %(self.conf_dict)
            #replacing {[key]} placeholders
            command = command.format(**self.conf_dict )
        except KeyError as exc:
            self.logger.error( "Unresolved template key %s in the command." %str(exc) )
            return 3
        self.logger.info('Command: "%s"', command )
        try:
            self.logger.info('Executing command...' )
            pipe = subprocess.Popen( command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT )
            out = ''
            for line in iter(pipe.stdout.readline, ''):
                if args.verbose>=1:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                out += line
            pipe.communicate()
            self.logger.info( 'Command returned code: %s' %pipe.returncode )
            ret = pipe.returncode
        except Exception as e:
            self.logger.error( str(e) )
            return 4
        ended = self.endJob( pipe.returncode, out )
        if ended != 0:
            ret = ended
        with open(logFile,'a') as logF:
            logF.write(out)
        return ret
    
import optparse
import argparse

class O2OTool():

    def execute(self):
        parser = argparse.ArgumentParser(description='CMS o2o command-line tool. For general help (manual page), use the help subcommand.')
        parser.add_argument('--db', type=str, help='The target database: pro ( for prod ) or dev ( for prep ). default=pro')
        parser.add_argument("--auth","-a", type=str,  help="The path of the authentication file")
        parser.add_argument('--verbose', '-v', action='count', help='The verbosity level')
        parser_subparsers = parser.add_subparsers(title='Available subcommands')
        parser_create = parser_subparsers.add_parser('create', description='Create a new O2O job')
        parser_create.add_argument('--name', '-n', type=str, help='The o2o job name',required=True)
        parser_create.add_argument('--configFile', '-c', type=str, help='the JSON configuration file path',required=True)
        parser_create.add_argument('--interval', '-i', type=int, help='the chron job interval',default=0)
        parser_create.set_defaults(func=self.create)
        parser_setConfig = parser_subparsers.add_parser('setConfig', description='Set a new configuration for the specified job. The configuration is expected as a list of entries "param": "value" (dictionary). The "param" labels will be used to inject the values in the command to execute. The dictionary is stored in JSON format.')
        parser_setConfig.add_argument('--name', '-n', type=str, help='The o2o job name',required=True)
        parser_setConfig.add_argument('--configFile', '-c', type=str, help='the JSON configuration file path',required=True)
        parser_setConfig.set_defaults(func=self.setConfig)
        parser_setInterval = parser_subparsers.add_parser('setInterval',description='Set a new execution interval for the specified job')
        parser_setInterval.add_argument('--name', '-n', type=str, help='The o2o job name',required=True)
        parser_setInterval.add_argument('--interval', '-i', type=int, help='the chron job interval',required=True)
        parser_setInterval.set_defaults(func=self.setInterval)
        parser_enable = parser_subparsers.add_parser('enable',description='enable the O2O job')
        parser_enable.add_argument('--name', '-n', type=str, help='The o2o job name',required=True)
        parser_enable.set_defaults(func=self.enable)
        parser_disable = parser_subparsers.add_parser('disable',description='disable the O2O job')
        parser_disable.add_argument('--name', '-n', type=str, help='The o2o job name',required=True)
        parser_disable.set_defaults(func=self.disable)
        parser_migrateConf = parser_subparsers.add_parser('migrateConfig',description='migrate the tag info for the jobs in configuration entries')
        parser_migrateConf.set_defaults(func=self.migrate)
        parser_listJobs = parser_subparsers.add_parser('listJobs', description='list the registered jobs')
        parser_listJobs.set_defaults(func=self.listJobs)
        parser_listConf = parser_subparsers.add_parser('listConfig', description='shows the configurations for the specified job')
        parser_listConf.add_argument('--name', '-n', type=str, help='The o2o job name',required=True)
        parser_listConf.add_argument('--dump', type=int, help='Dump the specified config.',default=0)
        parser_listConf.set_defaults(func=self.listConf)
        parser_dumpConf = parser_subparsers.add_parser('dumpConfig', description='dumps a specific job configuration version')
        parser_dumpConf.add_argument('versionIndex', type=str,help='the version to dump')
        parser_dumpConf.add_argument('--name', '-n', type=str, help='The o2o job name',required=True)
        parser_dumpConf.add_argument('--configFile', '-c', type=str, help='the JSON configuration file name - default:[jobname]_[version].json')
        parser_dumpConf.set_defaults(func=self.dumpConf)
        parser_run = parser_subparsers.add_parser('run', description='Wrapper for O2O jobs execution. Supports input parameter injection from the configuration file associated to the job. The formatting syntax supported are the python ones: "command -paramName {paramLabel}" or "command -paramName %(paramLabel)s". where [paramName] is the name of the parameter required for the command, and [paramLabel] is the key of the parameter entry in the config dictionary (recommended to be equal for clarity!"')
        parser_run.add_argument('executable', type=str,help='command to execute')
        parser_run.add_argument('--name', '-n', type=str, help='The o2o job name',required=True)
        parser_run.set_defaults(func=self.run)

        args = parser.parse_args()

        if args.verbose >=1:
            self.setup(args)
            return args.func()
        else:
            try:
                self.setup(args) 
                sys.exit( args.func())
            except Exception as e:
                logging.error(e)
                sys.exit(1)

    def setup(self, args):
        self.args = args
        db_service = prod_db_service
        if args.db is not None:
            if args.db == 'dev' or args.db == 'oradev' :
                db_service = dev_db_service
            elif args.db != 'orapro' and args.db != 'onlineorapro' and args.db != 'pro':
                raise Exception("Database '%s' is not known." %args.db )
        
        self.mgr = O2OJobMgr()
        return self.mgr.connect( db_service, args )
        
    def create(self):
        self.mgr.add( self.args.name, self.args.configFile, self.args.interval, True )

    def setConfig(self):
        self.mgr.setConfig( self.args.name, self.args.configFile )

    def setInterval(self):
        self.mgr.setInterval( self.args.name, self.args.interval )

    def enable(self):
        self.mgr.set( self.args.name, True )
    
    def disable(self):
        self.mgr.set( self.args.name, False )

    def migrate(self):
        self.mgr.migrateConfig()

    def listJobs(self):
        self.mgr.listJobs()

    def listConf(self):
        self.mgr.listConfig( self.args.name )

    def dumpConf(self):
        self.mgr.dumpConfig( self.args.name, self.args.versionIndex, self.args.configFile )

    def run(self):
        rmgr = self.mgr.runManager()
        return rmgr.executeJob( self.args )
