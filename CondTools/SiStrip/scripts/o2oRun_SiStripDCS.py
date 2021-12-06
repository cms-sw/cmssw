#!/usr/bin/env python3
'''
Top level script to run SiStrip DCS O2O.
@author: Huilin Qu
'''

import os
import sys
import atexit
import logging
import argparse
import subprocess
import traceback
import json
from functools import partial

import CondTools.SiStrip.o2o_helper as helper

logDirVar = 'O2O_LOG_FOLDER'

def run(args):
    logging.debug(args)

    is_ok = True

    o2ocmd = "SiStripDCSPopCon.py"
    o2ocmd += ' --delay {delay}'
    o2ocmd += ' --destTags {destTags}'
    o2ocmd += ' --destDb {destDb}'
    o2ocmd += ' --inputTag {inputTag}'
    o2ocmd += ' --sourceDb {sourceDb}'
    o2ocmd += ' --condDbRead {condDbRead}'
    if args.debug:
        o2ocmd += ' --debug'

    cmd = 'o2o --db {db} -v run -n {jobname} "{o2ocmd}"'.format(db=args.db, jobname=args.jobname, o2ocmd=o2ocmd)
    logging.info('Start running command:\n %s' % cmd)

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    atexit.register(partial(helper.kill_subproc_noexcept, p))

    log = p.communicate()[0]
    if p.returncode == 0:
        logging.info('O2OJob %s finished successfully!' % args.jobname)
    else:
        logging.error('O2OJob %s FAILED!' % args.jobname)
        is_ok = False

    return is_ok

def summary(args, is_ok, logfile):
    if is_ok:
        return

    # send the detailed log if failed
    debugLabel = '[TEST] ' if args.debug else ''
    with open(logfile, 'rb') as log:
        helper.send_mail(subject='%sDCS O2O Failure: %s' % (debugLabel, args.jobname),
                 message=log.read(),
                 send_to=args.mail_log_to,
                 send_from=args.mail_from)


def main():
    parser = argparse.ArgumentParser(description='Run SiStrip DCS O2O.')
    parser.add_argument('jobname', metavar='JOBNAME', type=str, help='O2O job name as in DB.')
    parser.add_argument('--mail-from', default='trk.o2o@cern.ch', help='Account to send email notification.')
    parser.add_argument('--mail-to', default='trk.o2o@cern.ch', help='List of O2O notification recipients.')
    parser.add_argument('--mail-log-to', default='trk.o2o@cern.ch', help='List of O2O log recipients.')
    parser.add_argument('--db', default='pro', help='The database for o2o job management: pro ( for prod ) or dev ( for prep ). Default: %(default)s.')
    parser.add_argument('--debug', action="store_true", default=False, help='Switch on debug mode. Default: %(default)s.')

    args = parser.parse_args()
    args.mail_to = args.mail_to.strip().split(',')
    args.mail_log_to = args.mail_log_to.strip().split(',')

    # Should NOT use logging before it's set up
    try:
        logdir = os.environ[logDirVar] if logDirVar in os.environ else '/tmp'
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        logfile = os.path.join(logdir, 'SiStripsDCSO2O_%s.log' % str(args.jobname))
        loglevel = logging.DEBUG if args.debug else logging.INFO
        helper.configLogger(logfile, loglevel)
    except Exception:
        # in case we failed before logging is set up
        # print the error, send an email, and exit
        helper.send_mail('DCS O2O Failure: %s' % args.jobname, traceback.format_exc(), args.mail_to, args.mail_from)
        raise

    try:
        is_ok = run(args)
        summary(args, is_ok, logfile)
    except Exception:
        # in case we failed before logging is set up
        # print the error, send an email, and exit
        helper.send_mail('DCS O2O Failure: %s' % args.jobname, traceback.format_exc(), args.mail_to, args.mail_from)
        raise

    if not is_ok:
        return ' --- O2O FAILED! ---'

if __name__ == '__main__':
    sys.exit(main())
