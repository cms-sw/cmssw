#!/usr/bin/env python3

from __future__ import print_function
import cx_Oracle
import datetime
import calendar
import sys
import logging
import CondCore.Utilities.conddb_serialization_metadata as sm
import CondCore.Utilities.credentials as auth
import CondCore.Utilities.conddb_time as conddb_time
import os

prod_db_service = ('cms_orcon_prod',{'w':'cms_orcon_prod/cms_cond_general_w','r':'cms_orcon_prod/cms_cond_general_r'})
adg_db_service = ('cms_orcon_adg',{'r':'cms_orcon_adg/cms_cond_general_r'})
dev_db_service = ('cms_orcoff_prep',{'w':'cms_orcoff_prep/cms_cond_general_w','r':'cms_orcoff_prep/cms_cond_general_r'})
schema_name = 'CMS_CONDITIONS'

fmt_str = "[%(asctime)s] %(levelname)s: %(message)s"
logLevel = logging.INFO
logFormatter = logging.Formatter(fmt_str)

def print_table( headers, table ):
    ws = []
    for h in headers:
        ws.append(len(h))
    for row in table:
        ind = 0
        for c in row:
            c = str(c)
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
        print(line)
    printf( headers )
    hsep = ''
    for w in ws:
        fmt = '{:-<%s}' %w
        hsep += (fmt.format('')+' ')
    print(hsep)
    for row in table:
        printf( row )

class version_db(object):
    def __init__(self, db ):
        self.db = db
        self.cmssw_boost_map = {}
        self.boost_run_map = []

    def fetch_cmssw_boost_map( self ):
        cursor = self.db.cursor()
        cursor.execute('SELECT BOOST_VERSION, CMSSW_VERSION FROM CMSSW_BOOST_MAP');
        rows = cursor.fetchall()
        self.cmssw_boost_map = {}
        for r in rows:
            self.cmssw_boost_map[r[1]]=r[0]
        return self.cmssw_boost_map   

    def fetch_boost_run_map( self ):
        cursor = self.db.cursor()
        cursor.execute('SELECT RUN_NUMBER, RUN_START_TIME, BOOST_VERSION, INSERTION_TIME FROM BOOST_RUN_MAP ORDER BY RUN_NUMBER, INSERTION_TIME')
        rows = cursor.fetchall()
        self.boost_run_map = []
        for r in rows:
            self.boost_run_map.append( (r[0],r[1],r[2],str(r[3])) )
        return self.boost_run_map

    def insert_boost_run_range( self, run, boost_version, min_ts ):
        cursor = self.db.cursor()
        cursor.execute('SELECT MIN(RUN_NUMBER) FROM RUN_INFO WHERE RUN_NUMBER >= :RUN',(run,))
        res = cursor.fetchone()
        if res is not None and res[0] is not None:
            min_run = res[0]
            cursor.execute('SELECT START_TIME FROM RUN_INFO WHERE RUN_NUMBER=:RUN',(min_run,))
            min_run_time = cursor.fetchone()[0]
            min_run_ts = calendar.timegm( min_run_time.utctimetuple() ) << 32
        else:
            min_run = run
            min_run_ts = conddb_time.string_to_timestamp(min_ts)
        now = datetime.datetime.utcnow()
        cursor.execute('INSERT INTO BOOST_RUN_MAP ( RUN_NUMBER, RUN_START_TIME, BOOST_VERSION, INSERTION_TIME ) VALUES (:RUN, :RUN_START_T, :BOOST, :TIME)',(run,min_run_ts,boost_version,now) )

    def insert_cmssw_boost( self, cmssw_version,boost_version ):
        cursor = self.db.cursor()
        cursor.execute('INSERT INTO CMSSW_BOOST_MAP ( CMSSW_VERSION, BOOST_VERSION ) VALUES ( :CMSSW_VERSION, :BOOST_VERSION )',(cmssw_version,boost_version))

    def lookup_boost_in_cmssw( self, cmssw_version ):
        cmssw_v = sm.check_cmssw_version(  cmssw_version )
        the_arch = None
        releaseRoot = None
        if sm.is_release_cycle( cmssw_v ):
            cmssw_v = sm.strip_cmssw_version( cmssw_v )
            archs = sm.get_production_arch( cmssw_v )
            for arch in archs:
                path = sm.get_release_root( cmssw_v, arch )
                if os.path.exists(os.path.join(path,cmssw_v)):
                    releaseRoot = path
                    the_arch = arch
                    break
            if releaseRoot is None:
                for arch in archs:
                    the_arch = arch
                    releaseRoot = sm.get_release_root( cmssw_v, arch )    
                    for r in sorted (os.listdir( releaseRoot )):
                        if r.startswith(cmssw_v):
                            cmssw_v = r
        logging.debug('Boost version will be verified in release %s' %cmssw_v) 

        if cmssw_v in self.cmssw_boost_map.keys():
            return self.cmssw_boost_map[cmssw_v]
    
        if releaseRoot is None:
            archs = sm.get_production_arch( cmssw_v )
            for arch in archs:
                path = sm.get_release_root( cmssw_v, arch )
                if os.path.exists(os.path.join(path,cmssw_v)):
                    releaseRoot = path
                    the_arch = arch
                    break
        logging.debug('Release path: %s' %releaseRoot)
        boost_version = sm.get_cmssw_boost( the_arch, '%s/%s' %(releaseRoot,cmssw_v) )
        if not boost_version is None:
            self.cmssw_boost_map[cmssw_v] = boost_version 
            self.insert_cmssw_boost( cmssw_v,boost_version )
        return boost_version

    def populate_for_gts( self ):
        cursor = self.db.cursor()
        cursor.execute('SELECT DISTINCT(RELEASE) FROM GLOBAL_TAG')
        rows = cursor.fetchall()
        for r in rows:
            self.lookup_boost_in_cmssw( r[0] )

class conddb_tool(object):
    def __init__( self ):
        self.db = None
        self.version_db = None
        self.args = None
        self.logger = logging.getLogger()        
        self.logger.setLevel(logLevel)
        consoleHandler = logging.StreamHandler(sys.stdout) 
        consoleHandler.setFormatter(logFormatter)
        self.logger.addHandler(consoleHandler)
        self.iovs = None
        self.versionIovs = None

    def connect( self ):
        if self.args.db is None:
            self.args.db = 'pro'
        if self.args.db == 'dev' or self.args.db == 'oradev' :
            db_service = dev_db_service
        elif self.args.db == 'orapro':
            db_service = adg_db_service    
        elif self.args.db != 'onlineorapro' or self.args.db != 'pro':
            db_service = prod_db_service
        else:
            raise Exception("Database '%s' is not known." %args.db )
        if self.args.accessType not in db_service[1].keys():
            raise Exception('The specified database connection %s does not support the requested action.' %db_service[0])
        service = db_service[1][self.args.accessType]
        creds = auth.get_credentials( service, self.args.auth )
        if creds is None:
            raise Exception("Could not find credentials for service %s" %service)
        (username, account, pwd) = creds
        connStr =  '%s/%s@%s' %(username,pwd,db_service[0])
        self.db = cx_Oracle.connect(connStr)
        logging.info('Connected to %s as user %s' %(db_service[0],username))
        self.db.current_schema = schema_name

    def process_tag_boost_version( self, t, timetype, tagBoostVersion, minIov, timeCut, validate ):
        if self.iovs is None:
            self.iovs = []
            cursor = self.db.cursor()
            stmt = 'SELECT IOV.SINCE SINCE, IOV.INSERTION_TIME INSERTION_TIME, P.STREAMER_INFO STREAMER_INFO FROM TAG, IOV, PAYLOAD P WHERE TAG.NAME = IOV.TAG_NAME AND P.HASH = IOV.PAYLOAD_HASH AND TAG.NAME = :TAG_NAME'
            params = (t,)
            if timeCut and tagBoostVersion is not None and not validate:
                whereClauseOnSince = ' AND IOV.INSERTION_TIME>:TIME_CUT' 
                stmt = stmt +  whereClauseOnSince
                params = params + (timeCut,)               
            stmt = stmt + ' ORDER BY SINCE'
            logging.debug('Executing: "%s"' %stmt)
            cursor.execute(stmt,params)
            for r in cursor:
                streamer_info = str(r[2].read())
                self.iovs.append((r[0],r[1],streamer_info))
        niovs = 0
        self.versionIovs = []
        lastBoost = None
        update = False
        if tagBoostVersion is not None:
            update = True        
        for iov in self.iovs:            
            if validate and timeCut is not None and  timeCut < iov[1]:
                continue
            niovs += 1
            iovBoostVersion, tagBoostVersion = sm.update_tag_boost_version( tagBoostVersion, minIov, iov[2], iov[0], timetype, self.version_db.boost_run_map )
            if minIov is None or iov[0]<minIov:
                minIov = iov[0]
            logging.debug('iov: %s - inserted on %s - streamer: %s' %(iov[0],iov[1],iov[2]))
            logging.debug('current tag boost version: %s minIov: %s' %(tagBoostVersion,minIov))
            if lastBoost is None or lastBoost!=iovBoostVersion:
                self.versionIovs.append((iov[0],iovBoostVersion))
                lastBoost = iovBoostVersion

        if tagBoostVersion is None:
            if niovs == 0:
                logging.warning( 'No iovs found. boost version cannot be determined.')
                return None, None
            else:
                logging.error('Could not determine the tag boost version.' )
                return None, None
        else:
            if niovs == 0:
                logging.info('Tag boost version has not changed.')
            else:
                msg = 'Found tag boost version %s ( min iov: %s ) combining payloads from %s iovs' %(tagBoostVersion,minIov,niovs)
                if timeCut is not None:
                    if update:
                        msg += ' (iov insertion time>%s)' %str(timeCut)
                    else:
                        msg += ' (iov insertion time<%s)' %str(timeCut)
                logging.info( msg ) 
        return tagBoostVersion, minIov

    def validate_boost_version( self, t, timetype, tagBoostVersion ):
        cursor = self.db.cursor()
        cursor.execute('SELECT GT.NAME, GT.RELEASE, GT.SNAPSHOT_TIME FROM GLOBAL_TAG GT, GLOBAL_TAG_MAP GTM WHERE GT.NAME = GTM.GLOBAL_TAG_NAME AND GTM.TAG_NAME = :TAG_NAME',(t,))
        rows = cursor.fetchall()
        invalid_gts = []
        ngt = 0
        gts = []
        for r in rows:
            gts.append((r[0],r[1],r[2]))
        if len(gts)>0:
            logging.info('validating %s gts.' %len(gts))
        boost_snapshot_map = {}
        for gt in gts:
            ngt += 1
            logging.debug('Validating for GT %s (release %s)' %(gt[0],gt[1])) 
            gtCMSSWVersion =  sm.check_cmssw_version( gt[1] )
            gtBoostVersion = self.version_db.lookup_boost_in_cmssw( gtCMSSWVersion  )
            if sm.cmp_boost_version( gtBoostVersion, tagBoostVersion )<0:
                logging.warning( 'The boost version computed from all the iovs in the tag (%s) is incompatible with the gt [%s] %s (consuming ver: %s, snapshot: %s)' %(tagBoostVersion,ngt,gt[0],gtBoostVersion,str(gt[2])))
                if str(gt[2]) not in boost_snapshot_map.keys():
                    tagSnapshotBoostVersion = None
                    minIov = None
                    tagSnapshotBoostVersion, minIov = self.process_tag_boost_version(t, timetype, tagSnapshotBoostVersion, minIov, gt[2])
                    if tagSnapshotBoostVersion is not None:
                        boost_snapshot_map[str(gt[2])] = tagSnapshotBoostVersion
                    else:
                        continue
                else:
                    tagSnapshotBoostVersion = boost_snapshot_map[str(gt[2])] 
                if sm.cmp_boost_version( gtBoostVersion, tagSnapshotBoostVersion )<0:
                    logging.error('The snapshot from tag used by gt %s (consuming ver: %s) has an incompatible combined boost version %s' %(gt[0],gtBoostVersion,tagSnapshotBoostVersion))
                    invalid_gts.append( ( gt[0], gtBoostVersion ) )
        if len(invalid_gts)==0:
            if ngt>0:
                logging.info('boost version for the tag validated in %s referencing Gts' %(ngt))
            else:
                logging.info('No GT referencing this tag found.')
        else:
            logging.error( 'boost version for the tag is invalid.')
        return invalid_gts

    def update_tag_boost_version_in_db( self, t, tagBoostVersion, minIov, update ):
        cursor = self.db.cursor()
        now = datetime.datetime.utcnow()
        if update:
            cursor.execute('UPDATE TAG_METADATA SET MIN_SERIALIZATION_V=:BOOST_V, MIN_SINCE=:MIN_IOV, MODIFICATION_TIME=:NOW WHERE TAG_NAME = :NAME',( tagBoostVersion,minIov,now,t))
        else:
            cursor.execute('INSERT INTO TAG_METADATA ( TAG_NAME, MIN_SERIALIZATION_V, MIN_SINCE, MODIFICATION_TIME ) VALUES ( :NAME, :BOOST_V, :MIN_IOV, :NOW )',(t, tagBoostVersion,minIov,now))
        logging.info('Minimum boost version for the tag updated.')
        
    def update_tags( self ):
        cursor = self.db.cursor()
        self.version_db = version_db( self.db )
        self.version_db.fetch_cmssw_boost_map()
        self.version_db.fetch_boost_run_map()
        tags = {}
        wpars = ()
        if self.args.name is not None:
            stmt0 = 'SELECT NAME FROM TAG WHERE NAME = :TAG_NAME'
            wpars = (self.args.name,)
            cursor.execute(stmt0,wpars);
            rows = cursor.fetchall()
            found = False
            for r in rows:
                found = True
                break
            if not found:
                raise Exception('Tag %s does not exists in the database.' %self.args.name )
            tags[self.args.name] = None
            stmt1 = 'SELECT MIN_SERIALIZATION_V, MIN_SINCE, CAST(MODIFICATION_TIME AS TIMESTAMP(0)) FROM  TAG_METADATA WHERE TAG_NAME = :NAME'
            cursor.execute(stmt1,wpars);
            rows = cursor.fetchall()
            for r in rows:
                tags[self.args.name] = (r[0],r[1],r[2])
        else:
            stmt0 = 'SELECT NAME FROM TAG WHERE NAME NOT IN ( SELECT TAG_NAME FROM TAG_METADATA) ORDER BY NAME' 
            nmax = 100
            if self.args.max is not None:
                nmax = self.args.max
            if self.args.all:
                nmax = -1
            if nmax >=0:
                stmt0 = 'SELECT NAME FROM (SELECT NAME FROM TAG WHERE NAME NOT IN ( SELECT TAG_NAME FROM TAG_METADATA ) ORDER BY NAME) WHERE ROWNUM<= :MAXR'
                wpars = (nmax,)
            cursor.execute(stmt0,wpars);
            rows = cursor.fetchall()
            for r in rows:
                tags[r[0]] = None
            stmt1 = 'SELECT T.NAME NAME, TM.MIN_SERIALIZATION_V MIN_SERIALIZATION_V, TM.MIN_SINCE MIN_SINCE, CAST(TM.MODIFICATION_TIME AS TIMESTAMP(0)) MODIFICATION_TIME FROM TAG T, TAG_METADATA TM WHERE T.NAME=TM.TAG_NAME AND CAST(TM.MODIFICATION_TIME AS TIMESTAMP(0)) < (SELECT MAX(INSERTION_TIME) FROM IOV WHERE IOV.TAG_NAME=TM.TAG_NAME) ORDER BY NAME' 
            nmax = nmax-len(tags)
            if nmax >=0:
                stmt1 = 'SELECT NAME, MIN_SERIALIZATION_V, MIN_SINCE, MODIFICATION_TIME FROM (SELECT T.NAME NAME, TM.MIN_SERIALIZATION_V MIN_SERIALIZATION_V, TM.MIN_SINCE MIN_SINCE, CAST(TM.MODIFICATION_TIME AS TIMESTAMP(0)) MODIFICATION_TIME FROM TAG T, TAG_METADATA TM WHERE T.NAME=TM.TAG_NAME AND CAST(TM.MODIFICATION_TIME AS TIMESTAMP(0)) < (SELECT MAX(INSERTION_TIME) FROM IOV WHERE IOV.TAG_NAME=TM.TAG_NAME) ORDER BY NAME) WHERE ROWNUM<= :MAXR'
                wpars = (nmax,)
            cursor.execute(stmt1,wpars);
            rows = cursor.fetchall()
            i = 0
            for r in rows:
                i += 1
                if nmax >=0 and i>nmax:
                    break
                tags[r[0]] = (r[1],r[2],r[3])
        logging.info( 'Processing boost version for %s tags' %len(tags))
        count = 0
        for t in sorted(tags.keys()):
            count += 1
            try:
                update = False
                cursor.execute('SELECT TIME_TYPE FROM TAG WHERE NAME= :TAG_NAME',(t,))
                timetype = cursor.fetchone()[0]
                self.iovs = None
                logging.info('************************************************************************')
                logging.info('Tag [%s] %s - timetype: %s' %(count,t,timetype))
                tagBoostVersion = None
                minIov = None
                timeCut = None
                if tags[t] is not None:
                    update = True
                    tagBoostVersion = tags[t][0]
                    minIov = tags[t][1]
                    timeCut = tags[t][2]
                tagBoostVersion, minIov = self.process_tag_boost_version( t, timetype, tagBoostVersion, minIov, timeCut, self.args.validate )
                if tagBoostVersion is None:
                    continue
                logging.debug('boost versions in the %s iovs: %s' %(len(self.iovs),str(self.versionIovs)))
                if self.args.validate: 
                    invalid_gts = self.validate_boost_version( t, timetype, tagBoostVersion )
                    if len(invalid_gts)>0:
                        with open('invalid_tags_in_gts.txt','a') as error_file:
                            for gt in invalid_gts:
                                error_file.write('Tag %s (boost %s) is invalid for GT %s ( boost %s) \n' %(t,tagBoostVersion,gt[0],gt[1]))
                if len(self.iovs):
                    if self.iovs[0][0]<minIov:
                        minIov = self.iovs[0]
                self.update_tag_boost_version_in_db( t, tagBoostVersion, minIov, update )
                self.db.commit()
            except Exception as e:
                logging.error(str(e))

    def insert_boost_run( self ):
        cursor = self.db.cursor()
        self.version_db = version_db( self.db )
        if self.args.min_ts is None:
            raise Exception("Run %s has not been found in the database - please provide an explicit TimeType value with the min_ts parameter ." %self.args.since )
        self.version_db.insert_boost_run_range( self.args.since, self.args.label, self.args.min_ts )
        self.db.commit()
        logging.info('boost version %s inserted with since %s' %(self.args.label,self.args.since))

    def list_boost_run( self ):
        cursor = self.db.cursor()
        self.version_db = version_db( self.db )
        self.version_db.fetch_boost_run_map()
        headers = ['Run','Run start time','Boost Version','Insertion time']
        print_table( headers, self.version_db.boost_run_map ) 

    def show_tag_boost_version( self ):
        cursor = self.db.cursor()
        tag = self.args.tag_name
        cursor.execute('SELECT TIME_TYPE FROM TAG WHERE NAME= :TAG_NAME',(tag,))
        rows = cursor.fetchall()
        timeType = None
        t_modificationTime = None
        for r in rows:
            timeType = r[0]
        if timeType is None:
            raise Exception("Tag %s does not exist in the database." %tag)
        cursor.execute('SELECT MAX(INSERTION_TIME) FROM IOV WHERE TAG_NAME= :TAG_NAME',(tag,))
        rows = cursor.fetchall()
        for r in rows:
            t_modificationTime = r[0]
        if t_modificationTime is None:
            raise Exception("Tag %s does not have any iov stored." %tag)
        logging.info('Tag %s - timetype: %s' %(tag,timeType))
        cursor.execute('SELECT MIN_SERIALIZATION_V, MIN_SINCE, MODIFICATION_TIME FROM TAG_METADATA WHERE TAG_NAME= :TAG_NAME',(tag,))
        rows = cursor.fetchall()
        tagBoostVersion = None
        minIov = None
        v_modificationTime = None
        for r in rows:
            tagBoostVersion = r[0]
            minIov = r[1]
            v_modificationTime = r[2]
        if v_modificationTime is not None:
            if t_modificationTime > v_modificationTime:
                logging.warning('The minimum boost version stored is out of date.')
            else:
                logging.info('The minimum boost version stored is up to date.')
        mt = '-'
        if v_modificationTime is not None:
            mt = str(v_modificationTime)
        r_tagBoostVersion = None
        if self.args.rebuild or self.args.full:
            self.version_db = version_db( self.db )
            self.version_db.fetch_boost_run_map()
            timeCut = None
            logging.info('Calculating minimum boost version for the available iovs...')
            r_tagBoostVersion, r_minIov = self.process_tag_boost_version( tag, timeType, tagBoostVersion, minIov, timeCut )
        print('# Currently stored: %s (min iov:%s)' %(tagBoostVersion,minIov))
        print('# Last update: %s' %mt)
        print('# Last update on the iovs: %s' %str(t_modificationTime))
        if self.args.rebuild or self.args.full:
            print('# Based on the %s available IOVs: %s (min iov:%s)' %(len(self.iovs),r_tagBoostVersion,r_minIov))
            if self.args.full:
                headers = ['Run','Boost Version']
                print_table( headers, self.versionIovs ) 

import optparse
import argparse

def main():
    tool = conddb_tool()
    parser = argparse.ArgumentParser(description='CMS conddb command-line tool for serialiation metadata. For general help (manual page), use the help subcommand.')
    parser.add_argument('--db', type=str, help='The target database: pro ( for prod ) or dev ( for prep ). default=pro')
    parser.add_argument("--auth","-a", type=str,  help="The path of the authentication file")
    parser.add_argument('--verbose', '-v', action='count', help='The verbosity level')
    parser_subparsers = parser.add_subparsers(title='Available subcommands')
    parser_update_tags = parser_subparsers.add_parser('update_tags', description='Update the existing tag headers with the boost version')
    parser_update_tags.add_argument('--name', '-n', type=str, help='Name of the specific tag to process (default=None - in this case all of the tags will be processed.')
    parser_update_tags.add_argument('--max', '-m', type=int, help='the maximum number of tags processed',default=100)
    parser_update_tags.add_argument('--all',action='store_true', help='process all of the tags with boost_version = None')
    parser_update_tags.add_argument('--validate',action='store_true', help='validate the tag/boost version under processing')
    parser_update_tags.set_defaults(func=tool.update_tags,accessType='w')
    parser_insert_boost_version = parser_subparsers.add_parser('insert', description='Insert a new boost version range in the run map')
    parser_insert_boost_version.add_argument('--label', '-l',type=str, help='The boost version label',required=True)
    parser_insert_boost_version.add_argument('--since', '-s',type=int, help='The since validity (run number)',required=True)
    parser_insert_boost_version.add_argument('--min_ts', '-t',type=str, help='The since validity (Time timetype)', required=False)
    parser_insert_boost_version.set_defaults(func=tool.insert_boost_run,accessType='w')
    parser_list_boost_versions = parser_subparsers.add_parser('list', description='list the boost versions in the run map')
    parser_list_boost_versions.set_defaults(func=tool.list_boost_run,accessType='r') 
    parser_show_version = parser_subparsers.add_parser('show_tag', description='Display the minimum boost version for the specified tag (the value stored, by default)')
    parser_show_version.add_argument('tag_name',help='The name of the tag')
    parser_show_version.add_argument('--rebuild','-r',action='store_true',default=False,help='Re-calculate the minimum boost versio ')
    parser_show_version.add_argument('--full',action='store_true',default=False,help='Recalulate the minimum boost version, listing the versions in the iov sequence')
    parser_show_version.set_defaults(func=tool.show_tag_boost_version,accessType='r')
    args = parser.parse_args()
    tool.args = args
    if args.verbose is not None and args.verbose >=1:
        tool.logger.setLevel(logging.DEBUG)
        tool.connect()
        return args.func()
    else:
        try:
            tool.connect() 
            sys.exit( args.func())
        except Exception as e:
            logging.error(e)
            sys.exit(1)
    
if __name__ == '__main__':
    main()
