#!/usr/bin/env python

from datetime import datetime
from optparse import OptionParser

import sys
import os
import re
import pprint
import time

import eostools as castortools

class BatchManager:
    """
    This class manages batch jobs
    Used in batch scripts
    Colin Bernet 2008
    """

    # constructor
    # self is this
    # parse batch manager options
    def __init__(self):
        self.DefineOptions()


    def DefineOptions(self):
        # define options and arguments ====================================
        # how to add more doc to the help?
        self.parser_ = OptionParser()
        self.parser_.add_option("-o", "--output-dir", dest="outputDir",
                                help="Name of the local output directory for your jobs. This directory will be created automatically.",
                                default=None)
        self.parser_.add_option("-r", "--remote-copy", dest="remoteCopy",
                                help="remote output directory for your jobs. Example: /store/cmst3/user/cbern/CMG/HT/Run2011A-PromptReco-v1/AOD/PAT_CMG/RA2. This directory *must* be provided as a logical file name (LFN). When this option is used, all root files produced by a job are copied to the remote directory, and the job index is appended to the root file name. The Logger directory  will be sent back to the submision directory. For remote copy to PSI specify path like: '/pnfs/psi.ch/...'. Note: enviromental variable X509_USER_PROXY must point to home area before renewing proxy",
                                default=None)
        self.parser_.add_option("-f", "--force", action="store_true",
                                dest="force", default=False,
                                help="Don't ask any questions, just over-write")
        # this opt can be removed
        self.parser_.add_option("-n", "--negate", action="store_true",
                                dest="negate", default=False,
                                help="create jobs, but does not submit the jobs.")
        self.parser_.add_option("-b", "--batch", dest="batch",
                                help="batch command. default is: 'bsub -q 8nh < batchScript.sh'. You can also use 'nohup < ./batchScript.sh &' to run locally.",
                                default="bsub -q 8nh < ./batchScript.sh")
        self.parser_.add_option( "--option",
                                dest="extraOptions",
                                type="string",
                                action="append",
                                default=[],
                                help="Save one extra option (either a flag, or a key=value pair) that can be then accessed from the job config file")

    def ParseOptions(self):
        (self.options_,self.args_) = self.parser_.parse_args()
        if self.options_.remoteCopy == None:
            self.remoteOutputDir_ = ""
        else:
            # removing possible trailing slash
            self.remoteOutputDir_ = self.options_.remoteCopy.rstrip('/')
            if "psi.ch" in self.remoteOutputDir_: # T3 @ PSI:
                # overwriting protection to be improved
                if self.remoteOutputDir_.startswith("/pnfs/psi.ch"):
                    ld_lib_path = os.environ.get('LD_LIBRARY_PATH')
                    if ld_lib_path != "None":
                        os.environ['LD_LIBRARY_PATH'] = "/usr/lib64/:"+ld_lib_path  # to solve gfal conflict with CMSSW
                    os.system("gfal-mkdir srm://t3se01.psi.ch/"+self.remoteOutputDir_)
                    outputDir = self.options_.outputDir
                    if outputDir==None:
                        today = datetime.today()
                        outputDir = 'OutCmsBatch_%s' % today.strftime("%d%h%y_%H%M")
                    self.remoteOutputDir_+="/"+outputDir
                    os.system("gfal-mkdir srm://t3se01.psi.ch/"+self.remoteOutputDir_)
                    if ld_lib_path != "None":
                        os.environ['LD_LIBRARY_PATH'] = ld_lib_path  # back to original to avoid conflicts
                else:
                    print "remote directory must start with /pnfs/psi.ch to send to the tier3 at PSI"
                    print self.remoteOutputDir_, "not valid"
                    sys.exit(1)
            else: # assume EOS
                if not castortools.isLFN( self.remoteOutputDir_ ):
                    print 'When providing an output directory, you must give its LFN, starting by /store. You gave:'
                    print self.remoteOutputDir_
                    sys.exit(1)
                self.remoteOutputDir_ = castortools.lfnToEOS( self.remoteOutputDir_ )
                dirExist = castortools.isDirectory( self.remoteOutputDir_ )
                # nsls = 'nsls %s > /dev/null' % self.remoteOutputDir_
                # dirExist = os.system( nsls )
                if dirExist is False:
                    print 'creating ', self.remoteOutputDir_
                    if castortools.isEOSFile( self.remoteOutputDir_ ):
                        # the output directory is currently a file..
                        # need to remove it.
                        castortools.rm( self.remoteOutputDir_ )
                    castortools.createEOSDir( self.remoteOutputDir_ )
                else:
                    # directory exists.
                    if self.options_.negate is False and self.options_.force is False:
                        #COLIN need to reimplement protectedRemove in eostools
                        raise ValueError(  ' '.join(['directory ', self.remoteOutputDir_, ' already exists.']))
                    # if not castortools.protectedRemove( self.remoteOutputDir_, '.*root'):
                    # the user does not want to delete the root files

        self.remoteOutputFile_ = ""
        self.ManageOutputDir()
        return (self.options_, self.args_)


    def PrepareJobs(self, listOfValues, listOfDirNames=None):
        print 'PREPARING JOBS ======== '
        self.listOfJobs_ = []

        if listOfDirNames is None:
            for value in listOfValues:
                self.PrepareJob( value )
        else:
            for value, name in zip( listOfValues, listOfDirNames):
                self.PrepareJob( value, name )
        print "list of jobs:"
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint( self.listOfJobs_)


    # create output dir, if necessary
    def ManageOutputDir( self ):

        #if the output dir is not specified, generate a name
        #else
        #test if the directory exists
        #if yes, returns

        outputDir = self.options_.outputDir

        if outputDir==None:
            today = datetime.today()
            outputDir = 'OutCmsBatch_%s' % today.strftime("%d%h%y_%H%M%S")
            print 'output directory not specified, using %s' % outputDir

        self.outputDir_ = os.path.abspath(outputDir)

        if( os.path.isdir(self.outputDir_) == True ):
            input = ''
            if not self.options_.force:
                while input != 'y' and input != 'n':
                    input = raw_input( 'The directory ' + self.outputDir_ + ' exists. Are you sure you want to continue? its contents will be overwritten [y/n] ' )
            if input == 'n':
                sys.exit(1)
            else:
                os.system( 'rm -rf ' + self.outputDir_)

        self.mkdir( self.outputDir_ )


    def PrepareJob( self, value, dirname=None):
        '''Prepare a job for a given value.

        calls PrepareJobUser, which should be overloaded by the user.
        '''
        print 'PrepareJob : %s' % value
        dname = dirname
        if dname  is None:
            dname = 'Job_{value}'.format( value=value )
        jobDir = '/'.join( [self.outputDir_, dname])
        print '\t',jobDir
        self.mkdir( jobDir )
        self.listOfJobs_.append( jobDir )
        self.PrepareJobUser( jobDir, value )

    def PrepareJobUser(self, value ):
        '''Hook allowing user to define how one of his jobs should be prepared.'''
        print '\to be customized'


    def SubmitJobs( self, waitingTimeInSec=0 ):
        '''Submit all jobs. Possibly wait between each job'''

        if(self.options_.negate):
            print '*NOT* SUBMITTING JOBS - exit '
            return
        print 'SUBMITTING JOBS ======== '
        for jobDir  in self.listOfJobs_:
            root = os.getcwd()
            # run it
            print 'processing ', jobDir
            os.chdir( jobDir )
            self.SubmitJob( jobDir )
            # and come back
            os.chdir(root)
            print 'waiting %s seconds...' % waitingTimeInSec
            time.sleep( waitingTimeInSec )
            print 'done.'

    def SubmitJob( self, jobDir ):
        '''Hook for job submission.'''
        print 'submitting (to be customized): ', jobDir
        os.system( self.options_.batch )


    def SubmitJobArray( self, numbOfJobs = 1 ):
        '''Hook for array job submission.'''
        print 'Submitting array with %s jobs'  % numbOfJobs

    def CheckBatchScript( self, batchScript ):

        if batchScript == '':
            return

        if( os.path.isfile(batchScript)== False ):
            print 'file ',batchScript,' does not exist'
            sys.exit(3)

        try:
            ifile = open(batchScript)
        except:
            print 'cannot open input %s' % batchScript
            sys.exit(3)
        else:
            for line in ifile:
                p = re.compile("\s*cp.*\$jobdir\s+(\S+)$");
                m=p.match(line)
                if m:
                    if os.path.isdir( os.path.expandvars(m.group(1)) ):
                        print 'output directory ',  m.group(1), 'already exists!'
                        print 'exiting'
                        sys.exit(2)
                    else:
                        if self.options_.negate==False:
                            os.mkdir( os.path.expandvars(m.group(1)) )
                        else:
                            print 'not making dir', self.options_.negate

    # create a directory
    def mkdir( self, dirname ):
        # there is probably a command for this in python
        mkdir = 'mkdir -p %s' % dirname
        ret = os.system( mkdir )
        if( ret != 0 ):
            print 'please remove or rename directory: ', dirname
            sys.exit(4)


    def RunningMode(self, batch):

        '''Return "LXPUS", "PSI", "NAF", "LOCAL", or None,

        "LXPLUS" : batch command is bsub, and logged on lxplus
        "PSI"    : batch command is qsub, and logged to t3uiXX
        "NAF"    : batch command is qsub, and logged on naf
        "IC"     : batch command is qsub, and logged on hep.ph.ic.ac.uk
        "LOCAL"  : batch command is nohup.

        In all other cases, a CmsBatchException is raised
        '''

        hostName = os.environ['HOSTNAME']

        onLxplus = hostName.startswith('lxplus')
        onPSI    = hostName.startswith('t3ui')
        onNAF =  hostName.startswith('naf')

        batchCmd = batch.split()[0]

        if batchCmd == 'bsub':
            if not onLxplus:
                err = 'Cannot run %s on %s' % (batchCmd, hostName)
                raise ValueError( err )
            else:
                print 'running on LSF : %s from %s' % (batchCmd, hostName)
                return 'LXPLUS'

        elif batchCmd == "qsub":
            if onPSI:
                print 'running on SGE : %s from %s' % (batchCmd, hostName)
                return 'PSI'
            elif onNAF:
                print 'running on NAF : %s from %s' % (batchCmd, hostName)
                return 'NAF'
            elif onIC:
                print 'running on IC : %s from %s' % (batchCmd, hostName)
                return 'IC'
            else:
                err = 'Cannot run %s on %s' % (batchCmd, hostName)
                raise ValueError( err )

        elif batchCmd == 'nohup' or batchCmd == './batchScript.sh':
            print 'running locally : %s on %s' % (batchCmd, hostName)
            return 'LOCAL'
        else:
            err = 'unknown batch command: X%sX' % batchCmd
            raise ValueError( err )
