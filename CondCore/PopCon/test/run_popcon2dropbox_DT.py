#!/usr/bin/env python
'''Script that directs the popcon output to the dropbox
'''

__author__ = 'Giacomo Govi'

import sys
from CondCore.Utilities import popcon2dropbox

cmssw_dir = '/data/cmssw'
release_dir = '/nfshome0/popcondev/popcon2dropbox'
release = 'CMSSW_7_2_0_pre6'
scram_arch = 'slc5_amd64_gcc481'
job_file = 'popcon2dropbox_job.py'
log_file = 'popcon.log'
dbox_backend = 'offline' 

def main():
    print popcon2dropbox.runO2O( cmssw_dir, release_dir, release, scram_arch, job_file, log_file, *sys.argv[1:] )
    popcon2dropbox.upload_to_dropbox( dbox_backend )

if __name__ == '__main__':
    sys.exit(main())


