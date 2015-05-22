import os
import subprocess
import unittest

import ROOT as rt

def cmsRun(cfgFile):
    """Run the specified cmsRun"""
    if not os.path.exists(cfgFile):
        raise IOError("The file '%s' does not exist" % cfgFile)
    return subprocess.Popen(['cmsRun',cfgFile], stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()

def getObject(rootFile, objectName):
    """Get an object from a Root file"""
    if not os.path.exists(rootFile):
        raise IOError("The file '%s' does not exist" % rootFile)

    obj = None
    try:
        input = rt.TFile.Open(rootFile)
        obj = input.Get(objectName)
    finally:
        pass
    return obj

def getEntries(rootFile, treeName = None):
    """Find out how many events are in a tree"""
    if treeName is None:
        treeName = 'Events'
    tree = getObject(rootFile, treeName)
    entries = -1
    if tree is not None:
        entries = tree.GetEntries()
    return entries

def getCfg(path):
    cfg = os.path.expandvars(os.path.join('$CMSSW_BASE','src',path))
    if not os.path.exists(cfg):
        raise IOError("The file '%s' does not exist" % cfg)
    return cfg

def parseCfg(path):
    l = {}
    g = {}
    execfile(path,g,l)
    return (l,g)

def getOutputFiles(path):

    l,g = parseCfg(path)
    tupleFile = l['process'].out.fileName.value()
    histFile = l['process'].TFileService.fileName.value()

    pwd = os.getcwd()

    return (os.path.join(pwd,tupleFile),os.path.join(pwd,histFile))

class CFGTest(unittest.TestCase):

    #this is a bit of a hack, but works OK
    _setUpOnce = False
    cfgsRunOnceCache = {}

    def __init__(self,methodName):
        unittest.TestCase.__init__(self, methodName)

        self.cfgs = []
        self.cfgsCache = {}

        self.cfgsRunOnce = []

    def setupOnce(self):
        """Like setUp but only called once per TestCase"""
        if self.__class__._setUpOnce:
            return
        self.__class__._setUpOnce = True
        for c in self.cfgsRunOnce:
            cfg = getCfg(c)
            tupleFile, histFile = getOutputFiles(cfg)
            stdout,stderr = cmsRun(cfg)

            self.__class__.cfgsRunOnceCache[c] = (stdout,tupleFile,histFile,stderr)

    def tearDownOnce(self):
        """Like tearDown, but only called at the end of all test cases"""
        #clean up the files
        for key, val in self.__class__.cfgsRunOnceCache.iteritems():
            try:
                os.remove(val[1])
                os.remove(val[2])
            except:
                pass
        self.__class__.cfgsRunOnceCache.clear()

    def __del__(self):
        self.tearDownOnce()

    def testSetupOnceFilesExist(self):
        """Tests that the files created by setUpOnce exist"""
        for key, val in self.__class__.cfgsRunOnceCache.iteritems():
            self.assertTrue(os.path.exists(val[1]),"The file '%s' is missing" % val[1])
            self.assertTrue(os.path.exists(val[2]),"The file '%s' is missing" % val[2])

    def testSetupFilesExist(self):
        """Tests that the files created by setUp exist"""
        for key, val in self.cfgsCache.iteritems():
            self.assertTrue(os.path.exists(val[1]),"The file '%s' is missing" % val[1])
            self.assertTrue(os.path.exists(val[2]),"The file '%s' is missing" % val[2])

    def testSetupOnceExceptions(self):
        """Looks in the stdout and stderr for the word 'Exception'"""
        for key, val in self.__class__.cfgsRunOnceCache.iteritems():
            self.assertFalse('Exception' in val[0],'The stdout should not have any exceptions')
            self.assertFalse('Exception' in val[3],'The stderr should not have any exceptions')

    def testSetupExceptions(self):
        """Looks in the stdout and stderr for the word 'Exception'"""
        for key, val in self.cfgsCache.iteritems():
            self.assertFalse('Exception' in val[0],'The stdout should not have any exceptions')
            self.assertFalse('Exception' in val[3],'The stderr should not have any exceptions')

    def setUp(self):
        self.setupOnce()
        for c in self.cfgs:
            cfg = getCfg(c)
            tupleFile, histFile = getOutputFiles(cfg)
            stdout,stderr = cmsRun(cfg)

            self.cfgsCache[c] = (stdout,tupleFile,histFile,stderr)

    def tearDown(self):
        for key, val in self.cfgsCache.iteritems():
            try:
                os.remove(val[1])
                os.remove(val[2])
            except Exception, e:
                print 'tearDown: Error cleaning up',e
        self.cfgsCache.clear()
