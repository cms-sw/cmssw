__author__ = 'Giacomo Govi'

import sqlalchemy
import sqlalchemy.ext.declarative
import subprocess
from datetime import datetime
import os
import sys
import logging
import string

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

class O2ORun(_Base):
    __tablename__      = 'O2O_RUN'
    __table_args__     = {'schema' : schema_name}
    job_name           = sqlalchemy.Column(sqlalchemy.String(100),    primary_key=True)
    start_time         = sqlalchemy.Column(sqlalchemy.TIMESTAMP,      primary_key=True)
    end_time           = sqlalchemy.Column(sqlalchemy.TIMESTAMP,      nullable=True)
    status_code        = sqlalchemy.Column(sqlalchemy.Integer,        nullable=False)
    log                = sqlalchemy.Column(sqlalchemy.CLOB,           nullable=True)

def get_db_credentials( db_service, authFile ):
    (username, account, pwd) = auth.get_credentials( authPathEnvVar, db_service[1], authFile )
    return username,pwd


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
            try:
                username, pwd = get_db_credentials( db_service, auth )
            except Exception as e:
                logging.debug(str(e))
                username = None
                pwd = None
            if username is None:
                logging.error('Credentials for service %s (machine=%s) are not available' %(db_service[0],db_service[1]))
                return None
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

    def connect( self, service, auth ):
        self.session = O2OMgr.getSession( self, service, auth )
        if self.session is None:
            return False
        else:
            return True

    def add( self, job_name, tag, int_val, en_flag ):
        res = self.session.query(O2OJob.enabled).filter_by(name=job_name)
        enabled = None
        for r in res:
            enabled = r
        if enabled:
            print 'ERROR: a job called %s exists already.' %job_name
            return False
        job = O2OJob(name=job_name,tag_name=tag,enabled=en_flag,interval=int_val)
        self.session.add(job)
        self.session.commit()
        return True

    def set( self, job_name, en_flag ):
        res = self.session.query(O2OJob.enabled).filter_by(name=job_name)
        enabled = None
        for r in res:
            enabled = r
        if not enabled:
            print 'ERROR: a job called %s does not exist.' %job_name
            return
        job = O2OJob(name=job_name,enabled=en_flag)
        self.session.merge(job)
        self.session.commit()

class O2ORunMgr(O2OMgr):

    def __init__( self ):
        O2OMgr.__init__(self)
        self.job_name = None
        self.start = None
        self.end = None
        self.tag_name = None
        self.db_connection = None

    def log( self, level, message ):
        consoleLog = getattr(O2OMgr.logger( self ),level)
        consoleLog( message )
        if self.logger:
            fileLog = getattr(self.logger, level )
            fileLog( message )

    def connect( self, service, args ):
        self.session = O2OMgr.getSession( self,service, args.auth )
        self.verbose = args.verbose
        if self.session is None:
            return False
        else:
            self.db_connection = coral_tpl %(service[0],schema_name)
            return True

    def startJob( self, job_name ):
        O2OMgr.logger( self ).info('Checking job %s', job_name)
        exists = None
        enabled = None
        try:
            res = self.session.query(O2OJob.enabled,O2OJob.tag_name).filter_by(name=job_name)
            for r in res:
                exists = True
                enabled = int(r[0])
                self.tag_name = str(r[1]) 
            if exists is None:
                exists = False
                enabled = False
            if enabled:
                self.job_name = job_name
                self.start = datetime.now()
                run = O2ORun(job_name=self.job_name,start_time=self.start,status_code=startStatus)
                self.session.add(run)
                self.session.commit()
        except sqlalchemy.exc.SQLAlchemyError as dberror:
                print 'Error: [%s]' %str(dberror) 
                O2OMgr.logger( self ).error( str(dberror) )
        return exists, enabled 


    def endJob( self, status, log ):
        self.end = datetime.now()
        try:
            run = O2ORun(job_name=self.job_name,start_time=self.start,end_time=self.end,status_code=status,log=log)
            self.session.merge(run)
            self.session.commit()
            O2OMgr.logger( self ).info( 'Job %s ended.', self.job_name )
        except sqlalchemy.exc.SQLAlchemyError as dberror:
            O2OMgr.logger( self ).error( str(dberror) )

    def executeJob( self, args ):
        job_name = args.name
        command = args.executable
        logFolder = os.getcwd()
        if logFolderEnvVar in os.environ:
            logFolder = os.environ[logFolderEnvVar]
        datelabel = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        logFileName = '%s-%s.log' %(job_name,datelabel)
        logFile = os.path.join(logFolder,logFileName)
        exists, enabled = self.startJob( job_name )
        if exists is None:
            return 3
        if not exists:
            O2OMgr.logger( self).error( 'The job %s is unknown.', job_name )
            return 2
        else:
            if enabled == 0:
                O2OMgr.logger( self).info( 'The job %s has been disabled.', job_name )
                return 5
        if args.inputFromDb:
            try:
                O2OMgr.logger( self ).info('Setting db input parameters...') 
                input_params = {'db':self.db_connection,'tag':self.tag_name }
                commandTpl = string.Template( command )
                command = commandTpl.substitute( input_params )
            except KeyError as exc:
                O2OMgr.logger( self).error( str(exc)+': Unknown template key in the command.' )
        O2OMgr.logger( self ).info('O2O Command: "%s"', command )
        try:
            O2OMgr.logger( self ).info('Executing job %s', job_name )
            pipe = subprocess.Popen( command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT )
            out = ''
            for line in iter(pipe.stdout.readline, ''):
                if self.verbose:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                out += line
            pipe.communicate()
            O2OMgr.logger( self ).info( 'Job %s returned code: %s' %(job_name,pipe.returncode) )
        except Exception as e:
            O2OMgr.logger( self ).error( str(e) )
            return 4
        self.endJob( pipe.returncode, out )
        with open(logFile,'a') as logF:
            logF.write(out)
        return pipe.returncode

    
