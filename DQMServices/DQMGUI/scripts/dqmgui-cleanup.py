#!/usr/bin/env python3
import os
import sys
import time
import argparse
import subprocess
from collections import defaultdict

def start(directory, cmsswSymlink):
    print('Starting to watch directory: %s' % directory)

    while True:
        time.sleep(10)

        try:
            allPBFiles = defaultdict(list)
            for file in os.listdir(directory):
                if not os.path.isfile(os.path.join(directory, file)):
                    continue
                if not file.startswith('run') and not file.endswith('.pb'):
                    continue

                try:
                    run = int(file[3:9])
                except:
                    print('Unable to get run number form a PB file: %s' % file)
                    continue

                allPBFiles[run].append(os.path.join(directory, file))

            # We will delete all but the newest run
            newestRun = max(allPBFiles.keys())

            for run in allPBFiles.keys():
                if run != newestRun:
                    for file in allPBFiles[run]:
                        try:
                            print('Removing: %s' % file)
                            os.remove(file)
                        except:
                            pass
        except Exception as ex:
            print(ex)

        # Check if CMSSW was updated
        cmsswDir = subprocess.check_output(['readlink', '-e', cmsswSymlink])
        cmsswDir = cmsswDir.decode('utf-8').rstrip()
        if os.getcwd() != cmsswDir:
            print('CMSSW release changed - exiting.')
            sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
        '''This utility removes old PB files from the Online DQM GUI machine.
        Only the latest file of the latest run is kept.''')
    parser.add_argument('-d', '--directory', default='/data/dqmgui/files/pb/', help='Directory from which PB files will be deleted.')
    parser.add_argument('-c', '--cmsswSymlink', default='/dqmdata/dqm_cmssw/current_production/src/', help='Symbolic link to the current CMSSW release.')
    args = parser.parse_args()

    start(args.directory, args.cmsswSymlink)
