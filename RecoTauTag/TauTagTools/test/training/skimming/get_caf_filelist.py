import subprocess
import sys
import os

caf_directory = sys.argv[1]
castor_dir = None

if caf_directory.startswith('/castor/cern.ch/cms'):
    caf_directory = caf_directory.replace('/castor/cern.ch/cms', '')

castor_dir = '/castor/cern.ch/cms'+caf_directory

p = subprocess.Popen(['nsls', castor_dir], stdout=subprocess.PIPE)

for line in p.stdout.readlines():
    print os.path.join(caf_directory, line.strip())
