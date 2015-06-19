from optparse import OptionParser
import sys,os, re, subprocess, datetime

import eostools as castortools

class logger:
    '''COLIN: do something cleaner with tagPackage'''
    def __init__(self, dirLocalOrTgzDirOnCastor):
        
        self.dirLocal = None
        self.tgzDirOnCastor = None
        dirLocalOrTgzDirOnCastor = dirLocalOrTgzDirOnCastor.rstrip('/')

        if self.isDirLocal( dirLocalOrTgzDirOnCastor ):
            self.dirLocal = dirLocalOrTgzDirOnCastor
        elif self.isTgzDirOnCastor( dirLocalOrTgzDirOnCastor ):
            self.tgzDirOnCastor = dirLocalOrTgzDirOnCastor
        else:
            raise ValueError( dirLocalOrTgzDirOnCastor + ' is neither a tgz directory on castor (provide a LFN!) nor a local directory')
            
        
    def isDirLocal(self, file ):
        if os.path.isdir( file ):
            return True
        else:
            return False

    def isTgzDirOnEOS(self, file ):
        '''Checks if file is a .tgz file in an eos dir'''
        if not castortools.isCastorDir( file ):
            file = castortools.castorToLFN(file)
            
        if castortools.isLFN( file ):
            tgzPattern = re.compile('.*\.tgz$')
            m = tgzPattern.match( file )
            if m:
                return True
            else:
                return False
        else:
            return False

    isTgzDirOnCastor = isTgzDirOnEOS

    def dump(self):
        print 'local dir      :', self.dirLocal
        print 'castor archive :',self.tgzDirOnCastor

    def addFile(self, file):
        #        if self.dirLocal == None:
        #            self.stageIn()
        #            os.system( 'cp %s %s' % (file, self.dirLocal) )
        #            self.stageOut( self.tgzDirOnCastor )
        if self.dirLocal != None:
            os.system( 'cp %s %s' % (file, self.dirLocal) )

    def logCMSSW(self): 
        showtagsLog = 'logger_showtags.txt'
        diffLog = 'logger_diff.txt'
        # os.system('showtags > ' + showtagsLog)
        self.showtags(showtagsLog)
        self.gitdiff(diffLog)
        self.addFile(showtagsLog)
        self.addFile(diffLog) 

    def logJobs(self, n):
        nJobs = 'logger_jobs.txt'
        out = file(nJobs,'w')
        out.write('NJobs: %i\n' % n)
        out.close()
        self.addFile(nJobs)

    def gitdiff(self, log):
        oldPwd = os.getcwd()
        os.chdir( os.getenv('CMSSW_BASE') + '/src/' )
        diffCmd = 'git diff -p --stat --color=never > %s/%s 2> /dev/null' % (oldPwd, log)
        print diffCmd
        os.system( diffCmd )
        os.chdir( oldPwd )

    def showtags(self, log):
        oldPwd = os.getcwd()
        os.chdir( os.getenv('CMSSW_BASE') + '/src/' )
        cmd = 'echo "Test Release based on: $CMSSW_VERSION" >  %s/%s 2> /dev/null' % (oldPwd, log)
        os.system( cmd )
        cmd = 'echo "Base Release in: $CMSSW_RELEASE_BASE"  >> %s/%s 2> /dev/null' % (oldPwd, log)
        os.system( cmd )
        cmd = 'echo "Your Test release in: $CMSSW_BASE"     >> %s/%s 2> /dev/null' % (oldPwd, log)
        os.system( cmd )
        cmd = 'git status --porcelain -b | head -n 1 >> %s/%s 2> /dev/null' % (oldPwd, log)
        os.system( cmd )
        cmd = 'git log -n 100 --format="%%T %%ai %%s %%d" >> %s/%s 2> /dev/null' % (oldPwd, log)
        os.system( cmd )
        os.chdir( oldPwd )        

    def stageIn(self):
        if self.tgzDirOnCastor != None:
            # castortools.xrdcp( '.', [self.tgzDirOnCastor] )
            cmsStage = 'cmsStage -f ' + self.tgzDirOnCastor + ' .'
            print cmsStage
            os.system( cmsStage ) 
            tgzDir = os.path.basename( self.tgzDirOnCastor )
            print tgzDir 
            os.system('tar -zxvf ' + tgzDir)
            os.system('rm ' + tgzDir )
            (root, ext) = os.path.splitext(tgzDir)
            self.dirLocal = root
        else:
            print 'cannot stage in, the log had not been staged out'

    def stageOut(self, castorDir):

        castorDir = castortools.eosToLFN( castorDir )
        if not castortools.isLFN( castorDir ):
            print 'cannot stage out, you need to provide an LFN as a destination directory, beginning with /store .'
            return False
        
        if self.dirLocal != None:
            tgzDir = self.dirLocal + '.tgz'
            tgzCmd = 'tar -zcvf ' + tgzDir + ' ' + self.dirLocal
            print tgzCmd
            os.system( tgzCmd)
            cmsStage = 'cmsStage -f %s %s' % (tgzDir, castorDir )
            print cmsStage
            os.system( cmsStage )
            os.system('rm ' + tgzDir )
            self.tgzDirOnCastor =  castorDir + '/' + tgzDir
        else:
            print 'cannot stage out, the log is not staged in'
            
