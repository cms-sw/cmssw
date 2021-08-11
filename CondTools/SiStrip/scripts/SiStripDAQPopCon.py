#!/usr/bin/env python3
'''Script that runs a single O2O for SiStrip DAQ.
@author: Huilin Qu
'''

import os
import atexit
import logging
import socket
import argparse
import subprocess
from functools import partial
import CondTools.SiStrip.o2o_helper as helper
from CondTools.SiStrip.o2o_db_cfgmap import DbManagerDAQ
from CondTools.SiStrip.o2o_db_gain import DbManagerGain

jobDirVar = 'JOBDIR'
cfg_template = 'CondTools/SiStrip/python/SiStripO2O_cfg_template.py'

def runjob(args):
    if args.debug:
        logging.debug(str(args))

    # read cfglines from input cfgfile
    with open(args.cfgfile) as cfgfile:
        cfglines = cfgfile.read()
        logging.debug(cfglines)

    # create config from template
    job_file = 'cfg_{type}_{run}.py'.format(type=args.analyzer, run=args.since)
    output_db = '{type}_{run}.db'.format(type=args.analyzer, run=args.since)
    if os.path.exists(output_db):
        logging.info('Output sqlite file %s already exists! Deleting...' % output_db)
        os.remove(output_db)
    hashmap_db = 'hashmap_{type}_{run}.db'.format(type=args.analyzer, run=args.since)
    if os.path.exists(hashmap_db):
        logging.info('Hashmap sqlite file %s already exists! Deleting...' % hashmap_db)
        os.remove(hashmap_db)
    replace_dict = {'_CFGLINES_' : cfglines.replace('\\', ''),
                    '_ANALYZER_' : args.analyzer,
                    '_USEANALYSIS_':'False',
                    '_CONDDB_'   : args.condDbRead,
                    '_DBFILE_'   : 'sqlite:///%s' % output_db,
                    '_TARGETTAG_': args.inputTag,
                    '_RUNNUMBER_': args.since,
                    '_HASHMAPDB_': args.hashmapDb,
                    '_MAPDBFILE_': 'sqlite:///%s' % hashmap_db,
                    '_SKIPPED_'  : '',
                    '_WHITELISTED_': '',
                    }
    if args.analyzer == 'SiStripO2OApvGain':
        # special treatment for G1 O2O
        skipped = ''
        if args.skiplistFile:
            with open(args.skiplistFile) as skipfile:
                skipped = skipfile.read()
        else:
            logging.warning('Skipped module list not provided! No module will be skipped...')
        whitelisted = ''
        if args.whitelistFile:
            with open(args.whitelistFile) as wfile:
                whitelisted = wfile.read()
        else:
            logging.warning('Module whitelist not provided!')
        replace_dict['_USEANALYSIS_'] = 'True'
        replace_dict['_SKIPPED_'] = skipped
        replace_dict['_WHITELISTED_'] = whitelisted
        replace_dict['_HASHMAPDB_'] = ''
        replace_dict['_MAPDBFILE_'] = ''

    # find template cfg file
    for basedir in os.environ['CMSSW_SEARCH_PATH'].split(':'):
        templatefile = os.path.join(basedir, cfg_template)
        if os.path.exists(templatefile):
            logging.info('Use template config file %s' % templatefile)
            break
    config = helper.insert_to_file(templatefile, job_file, replace_dict)
    logging.info('Start running O2O...')
    logging.debug(' ... config:\n%s\n' % config)

    # run cmssw job: raise error if failed
    command = 'cmsRun %s' % job_file
    pipe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    atexit.register(partial(helper.kill_subproc_noexcept, pipe))
    out = pipe.communicate()[0]
    logging.info('\n%s\n' % out)
    logging.info('@@@CMSSW job return code = %d@@@' % pipe.returncode)
    if pipe.returncode != 0:
        raise RuntimeError('O2O job FAILED!')

    # upload: raise error if failed
    if args.no_upload:
        logging.info('Will not run uploading as requested!')
        return
    if args.use_uploader:
        f = helper.upload_payload
    else:
        f = helper.copy_payload
    f(dbFile=output_db, inputTag=args.inputTag, destTags=args.destTags, destDb=args.destDb, since=args.since,
      userText='{type}, run: {run}'.format(type=args.analyzer, run=args.since))

    # post O2O tasks: bookkeeping for fast O2O or G1 O2O
    if args.analyzer == 'SiStripO2OApvGain':
        logging.info('Writting bookkeeping info to database.')
        dbmgr = DbManagerGain(args.bookkeeping_db)
        dbmgr.update_gain_logs(args.since, job_file)
    else:
        logging.info('Updating config-to-payload hash map to database.')
        dbmgr = DbManagerDAQ(args.bookkeeping_db)
        dbmgr.update_hashmap(hashmap_db)

    # clean up
    try:
        os.remove(output_db)
        os.remove(output_db.replace('.db', '.txt'))
        os.remove(hashmap_db)  # may not exist
    except OSError:
        pass

def main():
    parser = argparse.ArgumentParser(description='Run a single O2O job for SiStrip DAQ and upload the payloads to condition database.')
    parser.add_argument('analyzer', metavar='ANALYZER', help='Which EDAnalyzer to use to create the payload.')
    parser.add_argument('since', metavar='SINCE', type=str, help='Run number.')
    parser.add_argument('cfgfile', metavar='CFGLINES', help='File containing configuration lines.')
    parser.add_argument('--destTags', required=True, help='Destination tag name(s) for upload. Use comma to separate multiple values.')
    parser.add_argument('--destDb', required=True, help='Destination DB to upload.')
    parser.add_argument('--inputTag', required=True, help='Tag name to be used in the sqlite file.')
    parser.add_argument('--condDbRead', default='oracle://cms_orcon_prod/CMS_CONDITIONS', help='Connection string for the DB from which the fast O2O retrives payloads.')
    parser.add_argument('--hashmapDb', default='', help='DB to read and write config-to-payload hash (for fast O2O).')
    parser.add_argument('--skiplistFile', default='', help='File containing the devices to be skipped in G1 O2O.')
    parser.add_argument('--whitelistFile', default='', help='File of the whitelisted devices in G1 O2O.')

    parser.add_argument('--no-upload', action="store_true", default=False, help='Do not upload payload. Default: %(default)s.')
    parser.add_argument('--use-uploader', action="store_true", default=False, help='Use conditionUploader instead of conddb copy. Default: %(default)s.')
    parser.add_argument('--bookkeeping-db', default='prod', choices=['prod', 'dev', 'private'], help='Bookkeeping database for fast O2O and G1 O2O. Default: %(default)s.')
    parser.add_argument('--debug', action="store_true", default=False, help='Switch on debug mode. Default: %(default)s.')
    args = parser.parse_args()

    loglevel = logging.INFO
    if args.debug:
        loglevel = logging.DEBUG
        if args.bookkeeping_db == 'prod':
            args.bookkeeping_db = 'dev'
    logging.basicConfig(level=loglevel, format='[%(asctime)s] %(levelname)s: %(message)s')

    if not args.since.isdigit():
        raise RuntimeError('Since (=%s) must be a valid run number!'%(args.since))

    args.destTags = args.destTags.strip().split(',')

    logging.info('Running O2O %s on machine [%s]' % (args.analyzer, socket.gethostname()))

    try:
        jobdirbase = os.environ[jobDirVar]
    except KeyError:
        jobdirbase = '/tmp'
        logging.warning('%s not set in env, will use %s' % (jobDirVar, jobdirbase))

    # change filepaths in args to abs path
    args.cfgfile = os.path.abspath(args.cfgfile)
    if args.skiplistFile:
        args.skiplistFile = os.path.abspath(args.skiplistFile)
    if args.whitelistFile:
        args.whitelistFile = os.path.abspath(args.whitelistFile)

    # change to O2O working directory
    jobdir = os.path.join(jobdirbase, args.since, args.analyzer)
    if not os.path.exists(jobdir):
        os.makedirs(jobdir)
    os.chdir(jobdir)
    logging.info('Running O2O in %s' % jobdir)

    # run job and upload
    runjob(args)

    logging.info('Done!')



if __name__ == '__main__':
    main()
