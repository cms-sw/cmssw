#!/usr/bin/env python3
'''
Main Script to run all SiStrip DAQ O2Os at the same time.
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
    status = {}
    processes = {}

    for analyzer in args.analyzers:
        o2ocmd = 'SiStripDAQPopCon.py {analyzer} {since} {cfgfile}'.format(
            analyzer=analyzer, since=args.since, cfgfile=args.cfgfile)
        o2ocmd += ' --destTags {destTags}'
        o2ocmd += ' --destDb {destDb}'
        o2ocmd += ' --inputTag {inputTag}'
        o2ocmd += ' --condDbRead {condDbRead}'
        o2ocmd += ' --hashmapDb {hashmapDb}'
        if args.skiplistFile:
            o2ocmd += ' --skiplistFile %s' % args.skiplistFile
        if args.whitelistFile:
            o2ocmd += ' --whitelistFile %s' % args.whitelistFile
        if args.debug:
            o2ocmd += ' --debug'

        jobname = analyzer.replace('O2O', '')
        cmd = 'o2o --db {db} -v run -n {jobname} "{o2ocmd}"'.format(db=args.db, jobname=jobname, o2ocmd=o2ocmd)
        logging.info('Start running command:\n %s' % cmd)

        processes[jobname] = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        atexit.register(partial(helper.kill_subproc_noexcept, processes[jobname]))

    for jobname in processes:
        status[jobname] = {'job':None, 'upload':None, 'fast':None, 'changed':None}
        p = processes[jobname]
        log = p.communicate()[0]
        logging.debug('=== log from o2o run ===\n%s' % log)
        if p.returncode == 0:
            logging.info('Job for %s finished successfully!' % jobname)
            status[jobname]['job'] = True
            status[jobname]['upload'] = True
            for line in log.split('\n'):
                if '@@@' not in line:
                    continue
                if 'FastO2O' in line:
                    status[jobname]['fast'] = ('true' in line)
                if 'PayloadChange' in line:
                    status[jobname]['changed'] = ('true' in line)
        else:
            logging.error('Job %s FAILED!' % jobname)
            status[jobname]['job'] = '@@@CMSSW job return code = 0@@@' in log
            status[jobname]['upload'] = '@@@Upload return code = 0@@@' in log
            is_ok = False
            
    return is_ok, status

def summary(args, is_ok, status, logfile):
    summary = json.dumps(status, sort_keys=True, indent=2)
    if is_ok:
        logging.info('O2O finished successfully! Summary: %s' % summary)
    else:
        logging.error('O2O FAILED! Summary: %s' % summary)

    debugLabel = '[TEST] ' if args.debug else ''

    # send the summary email
    helper.send_mail(subject='%sNew O2O, IOV: %s' % (debugLabel, args.since),
             message=summary,
             send_to=args.mail_to,
             send_from=args.mail_from)
    # send the detailed log
    with open(logfile, 'rb') as log:
        helper.send_mail(subject='%sNew O2O Log, IOV: %s' % (debugLabel, args.since),
                 message=log.read(),
                 send_to=args.mail_log_to,
                 send_from=args.mail_from)


def main():
    parser = argparse.ArgumentParser(description='Run all SiStrip DAQ O2Os at the same time.')
    parser.add_argument('since', metavar='SINCE', type=str, help='Run number.')
    parser.add_argument('cfgfile', metavar='CFGLINES', help='File containing configuration lines.')
    parser.add_argument('--skiplistFile', default='', help='File containing the devices to be skipped in G1 O2O.')
    parser.add_argument('--whitelistFile', default='', help='File of the whitelisted devices in G1 O2O.')

    parser.add_argument('--analyzers',
                        default='SiStripO2OBadStrip,SiStripO2OFedCabling,SiStripO2OLatency,SiStripO2ONoises,SiStripO2OPedestals,SiStripO2OThreshold',
                        help='Which EDAnalyzers to run.')
    parser.add_argument('--mail-from', default='trk.o2o@cern.ch', help='Account to send email notification.')
    parser.add_argument('--mail-to', default='cms-tracker-o2o-notification@cern.ch', help='List of O2O notification recipients.')
    parser.add_argument('--mail-log-to', default='trk.o2o@cern.ch', help='List of O2O log recipients.')
    parser.add_argument('--db', default='pro', help='The database for o2o job management: pro ( for prod ) or dev ( for prep ). Default: %(default)s.')
    parser.add_argument('--debug', action="store_true", default=False, help='Switch on debug mode. Default: %(default)s.')

    args = parser.parse_args()
    if args.debug:
        args.mail_to = args.mail_log_to

    args.analyzers = args.analyzers.strip().split(',')
    args.mail_to = args.mail_to.strip().split(',')
    args.mail_log_to = args.mail_log_to.strip().split(',')

    # Should NOT use logging before it's set up
    try:
        logdir = os.environ[logDirVar] if logDirVar in os.environ else '/tmp'
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        logfile = os.path.join(logdir, 'SiStripsO2O_Run%s.log' % str(args.since))
        loglevel = logging.DEBUG if args.debug else logging.INFO
        helper.configLogger(logfile, loglevel)
    except Exception:
        # in case we failed before logging is set up
        # print the error, send an email, and exit
        helper.send_mail('O2O Failure, IOV: %s' % args.since, traceback.format_exc(), args.mail_to, args.mail_from)
        raise

    try:
        is_ok, status = run(args)
        summary(args, is_ok, status, logfile)
    except Exception:
        # in case we failed before logging is set up
        # print the error, send an email, and exit
        helper.send_mail('O2O Failure, IOV: %s' % args.since, traceback.format_exc(), args.mail_to, args.mail_from)
        raise

    if not is_ok:
        return ' --- O2O FAILED! ---'

if __name__ == '__main__':
    sys.exit(main())
