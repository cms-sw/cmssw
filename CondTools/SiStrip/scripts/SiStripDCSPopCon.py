#!/usr/bin/env python3
'''Script that runs SiStrip DCS O2O.
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

jobDirVar = 'JOBDIR'
cfg_file = 'CondTools/SiStrip/python/SiStripDCS_popcon.py'

def runjob(args):
    if args.debug:
        logging.debug(str(args))

    # find cfg file
    for basedir in os.environ['CMSSW_SEARCH_PATH'].split(':'):
        cfg = os.path.join(basedir, cfg_file)
        if os.path.exists(cfg):
            logging.info('Use config file %s' % cfg)
            break

    output_db = 'SiStripDetVOff_{delay}.db'.format(delay=args.delay)
    if os.path.exists(output_db):
        logging.info('Output sqlite file %s already exists! Deleting...' % output_db)
        os.remove(output_db)

    # run cmssw job: raise error if failed
    command = 'cmsRun {cfg} delay={delay} destinationConnection={destFile} sourceConnection={sourceDb} conddbConnection={conddb} tag={tag}'.format(
        cfg=cfg, delay=args.delay, destFile='sqlite:///%s' % output_db, sourceDb=args.sourceDb, conddb=args.condDbRead, tag=args.inputTag)
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

    if not helper.exists_iov(output_db, args.inputTag):
        logging.info('No IOV exists in the SQLite file. Will skip upload!')
        return

    if args.use_uploader:
        f = helper.upload_payload
    else:
        f = helper.copy_payload
    f(dbFile=output_db, inputTag=args.inputTag, destTags=args.destTags, destDb=args.destDb, since=None,
      userText='SiStripDCS {delay} hour delay'.format(delay=args.delay))

    # clean up
    try:
        os.remove(output_db)
        os.remove(output_db.replace('.db', '.txt'))
    except OSError:
        pass

def main():
    parser = argparse.ArgumentParser(description='Run a single O2O job for SiStrip DCS and upload the payloads to condition database.')
    parser.add_argument('--delay', required=True, help='Time delay (in hours) for the O2O. The O2O then queries the PVSS DB from last IOV until (current hour - delay), ignoring minutes and seconds.')
    parser.add_argument('--destTags', required=True, help='Destination tag name(s) for upload. Use comma to separate multiple values.')
    parser.add_argument('--sourceDb', required=True, help='Connection string for the source database.')
    parser.add_argument('--destDb', required=True, help='Destination DB to upload.')
    parser.add_argument('--inputTag', required=True, help='Tag name to be used in the sqlite file.')
    parser.add_argument('--condDbRead', default='oracle://cms_orcon_adg/CMS_CONDITIONS', help='Connection string for the DB from which the fast O2O retrives payloads.')

    parser.add_argument('--no-upload', action="store_true", default=False, help='Do not upload payload. Default: %(default)s.')
    parser.add_argument('--use-uploader', action="store_true", default=False, help='Use conditionUploader instead of conddb copy. Default: %(default)s.')
    parser.add_argument('--debug', action="store_true", default=False, help='Switch on debug mode. Default: %(default)s.')
    args = parser.parse_args()

    loglevel = logging.INFO
    if args.debug:
        loglevel = logging.DEBUG
    logging.basicConfig(level=loglevel, format='[%(asctime)s] %(levelname)s: %(message)s')

    args.destTags = args.destTags.strip().split(',')

    logging.info('Running DCS O2O with %s hour(s) delay on machine [%s]' % (str(args.delay), socket.gethostname()))

    try:
        jobdirbase = os.environ[jobDirVar]
    except KeyError:
        jobdirbase = '/tmp'
        logging.warning('%s not set in env, will use %s' % (jobDirVar, jobdirbase))


    # change to O2O working directory
    jobdir = os.path.join(jobdirbase, '{delay}hourDelay'.format(delay=args.delay))
    if not os.path.exists(jobdir):
        os.makedirs(jobdir)
    os.chdir(jobdir)
    logging.info('Running O2O in %s' % jobdir)

    # run job and upload
    runjob(args)

    logging.info('Done!')



if __name__ == '__main__':
    main()
