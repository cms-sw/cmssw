__author__ = 'Giacomo Govi'

import sqlalchemy
import sqlalchemy.ext.declarative
import subprocess
from datetime import datetime
import os
import sys
import netrc
import logging

prod_db_service = 'cms_orcon_prod'
dev_db_service = 'cms_orcoff_prep'
schema_name = 'CMS_CONDITIONS'
oracle_tpl = 'oracle://%s:%s@%s'
private_db = 'sqlite:///o2o_jobs.db'
startStatus = -1
authPathEnvVar = 'O2O_AUTH_PATH'
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
    name               = sqlalchemy.Column(sqlalchemy.String(100),    primary_key=True)
    enabled            = sqlalchemy.Column(sqlalchemy.Integer,        nullable=False)
    tag_name           = sqlalchemy.Column(sqlalchemy.String(100),    nullable=False)
    interval           = sqlalchemy.Column(sqlalchemy.Integer,        nullable=False)

class O2ORun(_Base):
    __tablename__      = 'O2O_RUN'
    job_name           = sqlalchemy.Column(sqlalchemy.String(100),    primary_key=True)
    start_time         = sqlalchemy.Column(sqlalchemy.TIMESTAMP,      primary_key=True)
    end_time           = sqlalchemy.Column(sqlalchemy.TIMESTAMP,      nullable=True)
    status_code        = sqlalchemy.Column(sqlalchemy.Integer,        nullable=False)
    log                = sqlalchemy.Column(sqlalchemy.CLOB,           nullable=True)

def get_db_credentials( serviceName, authFile ):
    pwd = None
    if authFile is None:
       if authPathEnvVar in os.environ:
            authPath = os.environ[authPathEnvVar]
            authFile = os.path.join(authPath,'.netrc')
            logging.debug('Retrieving credentials from file %s' %authFile )
    (username, account, pwd) = netrc.netrc( authFile ).authenticators(serviceName)
    return pwd


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
                pwd = get_db_credentials( db_service, auth )
            except Exception as e:
                logging.debug(str(e))
                pwd = None
            if not pwd:
                logging.error('Credentials for service %s are not available',db_service)
                return None
            url = oracle_tpl %(schema_name,pwd,db_service)
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

    def log( self, level, message ):
        consoleLog = getattr(O2OMgr.logger( self ),level)
        consoleLog( message )
        if self.logger:
            fileLog = getattr(self.logger, level )
            fileLog( message )

    def connect( self, service, auth ):
        self.session = O2OMgr.getSession( self,service, auth )
        if self.session is None:
            return False
        else:
            return True

    def startJob( self, job_name ):
        O2OMgr.logger( self ).info('Checking job %s', job_name)
        exists = None
        enabled = None
        try:
            res = self.session.query(O2OJob.enabled).filter_by(name=job_name)
            for r in res:
                exists = True
                enabled = r
            if exists is None:
                exists = False
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

    def executeJob( self, job_name, command ):
        logFolder = os.getcwd()
        if logFolderEnvVar in os.environ:
            logFolder = os.environ[logFolderEnvVar]
        datelabel = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        logFileName = '%s-%s.log' %(job_name,datelabel)
        logFile = os.path.join(logFolder,logFileName)
        exists, enabled = self.startJob( job_name )
        if exists is None:
            return 3
        if enabled is None:
            O2OMgr.logger( self).error( 'The job %s is unknown.', job_name )
            return 2
        else:
            if enabled == 0:
                O2OMgr.logger( self).error( 'The job %s has been disabled.', job_name )
                return 1
        try:
            O2OMgr.logger( self ).info('Executing job %s', job_name )
            pipe = subprocess.Popen( command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT )
            out = pipe.communicate()[0]
            O2OMgr.logger( self ).info( 'Job %s returned code: %s' %(job_name,pipe.returncode) )
        except Exception as e:
            O2OMgr.logger( self ).error( str(e) )
            return 4
        self.endJob( pipe.returncode, out )
        with open(logFile,'a') as logF:
            logF.write(out)
        return 0

    
