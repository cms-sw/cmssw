import unittest
import os

import CMGTools.Production.castortools as castortools
from CMGTools.Production.eostools import *

class TestEosTools(unittest.TestCase):
    
    def setUp(self):
        
        self.castorfile = '/castor/cern.ch/cms/store/cmst3/user/wreece/EOS_TEST/test_file.txt'
        self.eosfile = '/eos/cms/store/cmst3/user/wreece/EOS_TEST/test_file.txt'
        self.localfile = 'test_file.txt'
        self.eosdir = '/eos/cms/store/cmst3/user/%s/TmpEOS_Test' % os.environ['USER']
        self.localdir = 'EOS_Tests'
        
    def tearDown(self):
        # print 'calling tear down'
        if fileExists(self.eosfile + 'FOO'):
            rm(self.eosfile + 'FOO')
        if fileExists(self.eosfile + 'FOO2'):
            rm(self.eosfile + 'FOO2')
        if fileExists(self.localfile):
            rm( self.localfile)
        if fileExists(self.eosdir):
            rm( self.eosdir, rec=True)
        if fileExists(self.localdir):
            rm( self.localdir, rec=True)


    def checkDirsAndFiles(self, dir):
        filesAndDirs = listFiles( dir )
        print filesAndDirs

    def testCpFromEOS(self):
        xrdcp( '/store/cmst3/user/cbern/Tests/', self.localdir)

    def testCpToEOS(self):
        xrdcp( '/store/cmst3/user/cbern/Tests/', self.localdir)
        xrdcp( self.localdir, self.eosdir)
    
    def testWhich(self):
        
        self.assertEqual(which('cp'),'/bin/cp')
        self.assertEqual(which('ls'),'/bin/ls')
    
    def testIsLFN(self):
        
        self.assertFalse(isLFN(self.castorfile))
        self.assertFalse(isLFN(self.eosfile))
        
        lfn = eosToLFN(self.eosfile)
        self.assertTrue(isLFN(lfn))

    def testLFNToCastor(self):
        self.assertNotEqual(lfnToCastor(self.eosfile), castortools.lfnToCastor(self.castorfile))
        
    def testCastorToLFN(self):
        self.assertEqual(castorToLFN(self.eosfile), castortools.castorToLFN(self.castorfile))
    
    def testIsEOSDir(self):
        self.assertTrue(isEOSDir(eosToLFN(self.eosfile + 'FOO'))) #should still work if it doesn't exist
        self.assertTrue(isEOSDir(eosToLFN(self.eosfile)))
        self.assertTrue(isEOSDir(self.eosfile))
        self.assertFalse(isEOSDir(self.castorfile))
        
    def testIsEOSFile(self):
        self.assertTrue(isEOSFile(self.eosfile))
        self.assertFalse(isEOSFile(self.eosfile + 'FOO'))
        
    def testFileExists(self):
        self.assertTrue(fileExists(self.eosfile))
        self.assertFalse(fileExists(self.eosfile + 'FOO'))
        
        local = '/dev/null'
        self.assertTrue(fileExists(local))
        self.assertFalse(fileExists(local + 'FOO'))
        
        self.assertEqual(fileExists(local), castortools.fileExists(local))
        self.assertEqual(fileExists(local + 'FOO'), castortools.fileExists(local + 'FOO'))
        
    def testIsDirectory(self):
        
        d = os.path.dirname(self.eosfile)
        self.assertTrue(isDirectory(d))
        self.assertFalse(isDirectory(self.eosfile))
        
    def testIsFile(self):
        
        d = os.path.dirname(self.eosfile)
        self.assertFalse(isFile(d))
        self.assertTrue(isFile(self.eosfile))
        
    def testCreateEOSDir(self):
        
        d = os.path.join(os.path.dirname(self.eosfile), 'TEST_DIR', 'BAR', 'FOO')
        path = createEOSDir(d)
        self.assertTrue(isEOSDir(path))
        self.assertTrue(fileExists(path))
        
        self.assertTrue(isDirectory(path))
        
        rm(os.path.join(os.path.dirname(self.eosfile),'TEST_DIR'), rec = True)
        self.assertFalse(fileExists(path))
        
    def testChmod(self):
        
        _, _, ret = chmod(os.path.dirname(self.eosfile),'775')
        self.assertEquals(ret, 0)
        
    def testMatchingFiles(self):
        
        d = os.path.dirname(self.eosfile)
        matches = matchingFiles(d, '.*test_file\\.txt$')
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0], eosToLFN(self.eosfile))
        
    def testCat(self):
        xrdcp(self.eosfile, self.localfile)
        self.assertEqual(cat(self.eosfile), cat(self.localfile))
        self.assertEqual(cat(self.eosfile + 'FOO'), '')
        self.assertFalse('cat returned' in cat(self.eosfile))
        
    def testRemove(self):
        
        xrdcp(self.eosfile, self.eosfile + 'FOO')
        self.assertTrue( fileExists(self.eosfile + 'FOO'))
        remove([self.eosfile + 'FOO'])
        self.assertFalse( fileExists(self.eosfile + 'FOO'))
        
        xrdcp(self.eosfile, self.eosfile + 'FOO')
        self.assertTrue( fileExists(self.eosfile + 'FOO'))
        remove([self.eosfile + 'FOO'], rec = True)
        self.assertFalse( fileExists(self.eosfile + 'FOO'))
    
    def testCp(self):
        
        xrdcp(self.eosfile, self.eosfile + 'FOO')
        self.assertTrue( fileExists(self.eosfile + 'FOO'))
        rm(self.eosfile + 'FOO')
        
        import inspect
        this = inspect.getsourcefile(TestEosTools)
        
        d = os.path.dirname(self.eosfile)
        name = os.path.basename(this)
        xrdcp(this,d)
        self.assertTrue( fileExists(os.path.join(d,name)) )
        rm( os.path.join(d,name) )
        
        xrdcp(self.eosfile, os.getcwd())
        local = os.path.join(os.getcwd(), os.path.basename(self.eosfile))
        self.assertTrue( os.path.exists(local) )
        os.remove(local)
        
        xrdcp(self.eosfile, local)
        self.assertTrue( os.path.exists(local) )
        os.remove(local)            
        
    def testMove(self):
        
        xrdcp(self.eosfile, self.eosfile + 'FOO')
        self.assertTrue( fileExists(self.eosfile + 'FOO') )

        move(self.eosfile + 'FOO', self.eosfile + 'FOO2')
        self.assertTrue( fileExists(self.eosfile + 'FOO2'))
        self.assertFalse( fileExists(self.eosfile + 'FOO'))

        rm(self.eosfile + 'FOO')
        rm(self.eosfile + 'FOO2')
        
#            d = os.path.join(os.path.dirname(self.eosfile), 'TEST_DIR')
#            path = createEOSDir(d)
#            xrdcp(self.eosfile, path)
#            move(path, d+'2')
#            self.assertTrue( fileExists(d + '2'))
#            self.assertTrue( isDirectory(d + '2'))
if __name__ == "__main__":
    unittest.main()
    
