import sys
import os
import subprocess
from TkAlExceptions import AllInOneError

# script which needs to be sourced for use of crab
crabSourceScript = '/afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.sh'

# source the environment variables needed for crab
sourceStr = ( 'cd $CMSSW_BASE/src;'
              'source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.sh;'
              'eval `scramv1 runtime -sh`;'
              'source ' + crabSourceScript + ' && env' )
sourceCmd = ['bash', '-c', sourceStr ]
sourceProc = subprocess.Popen(sourceCmd, stdout = subprocess.PIPE)
for line in sourceProc.stdout:
    (key, _, value) = line.partition("=")
    os.environ[key] = value.replace("\n","")
sourceProc.communicate()

# source variables from crab wrapper script
crabFile = open('/'.join([os.environ["CRABPYTHON"],'crab']))
theLines = crabFile.readlines()
theLine = []
for line in theLines:
    if ( line[0] == '#' ) or \
           ( line == '  python $CRABPYTHON/crab.py $*\n' ):
        continue
    theLine.append( line )
tempFilePath = "tempCrab"
tempFile = open( tempFilePath, "w" )
tempFile.write( ''.join(theLine) )
tempFile.close()
crabStr = ('source tempCrab && env' )
crabCmd = ['bash', '-c', crabStr ]
crabProc = subprocess.Popen(crabCmd, stdout = subprocess.PIPE)
for line in crabProc.stdout:
    (key, _, value) = line.partition("=")
    os.environ[key] = value.replace("\n","")
crabProc.communicate()
os.remove( tempFilePath )

# add sourced paths to search path of python
sys.path.extend( os.environ["PYTHONPATH"].split( ':' ) )

import crab
import crab_exceptions

class CrabWrapper:
    def __init__( self ):
        pass

    def run( self, options ):
        theCrab = crab.Crab()
        theCrab.initialize_( options )
        try:
            theCrab.run()
        except crab_exceptions.CrabException, e:
            raise AllInOneError( str( e ) )
        del theCrab


if __name__ == "__main__":
    theCrab = CrabWrapper()
    theCrabOptions = {"-create":"",
                      "-cfg":"TkAlOfflineValidation.shiftPlots.crab.cfg"}
    theCrab.run( theCrabOptions )
    
    theCrabOptions = {"-submit":""}
    theCrab.run( theCrabOptions )
    
    theCrabOptions = {"-status":""}
    theCrab.run( theCrabOptions )

    theCrabOptions = {"-getoutput":""}
    try:
        theCrab.run( theCrabOptions )
    except AllInOneError, e:
        print "crab: ", e
