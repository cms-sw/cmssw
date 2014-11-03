import os
import re

def cmsswRelease():
    return os.environ['CMSSW_BASE'].split('/')[-1]

def cmsswIs44X():
    return cmsswRelease().find('CMSSW_4_4_') != -1

def cmsswIs52X():
    if cmsswRelease().find('CMSSW_5_2_') != -1 or \
            cmsswRelease().find('CMSSW_5_3_') != -1:
        return True
    else:
        return False

def releaseNumber(release = None):
    rerel = re.compile('^CMSSW_(\d+)_(\d+)_(\d+)(_\S+)*$')
    prel = re.compile('_patch(\d+)')
    if release is None:
        release = cmsswRelease()
    m = rerel.match(release)
    if m is None:
        raise ValueError('malformed release string '+release)
    big = int(m.group(1))
    medium = int(m.group(2))
    small = int(m.group(3))
    if m.group(4):
        pm = prel.match(m.group(4))
        if pm: 
            patch = int(pm.group(1))
            return big, medium, small, patch
        else:
            raise ValueError('patch string malformed '+m.group(4))
    else:
        return big, medium, small

def isNewerThan(release1, release2=None):
    """Checks the orders of two releases. If release2 is not set, it is taken as the current release"""
    return releaseNumber(release2) >= releaseNumber(release1)


if __name__ == '__main__':

    import unittest
    
    class CMSSWReleaseTest(unittest.TestCase):
        def test_cmsswRelease(self):
            rel = cmsswRelease()
            self.assertTrue(rel.startswith('CMSSW_'))
        def test_releaseNumber(self):
            out = releaseNumber('CMSSW_7_2_1')
            self.assertEqual(out, (7,2,1))
            out = releaseNumber('CMSSW_10_2_1_patch4')
            self.assertEqual(out, (10,2,1,4))
            self.assertRaises(ValueError, releaseNumber, 'foobar')
            self.assertRaises(ValueError, releaseNumber, 'CMSSW_1_2_3_xat3')
            self.assertRaises(ValueError, releaseNumber, 'CMSSW_1_2_a')
        def test_isNewerThan(self): 
            self.assertTrue( isNewerThan('CMSSW_5_3_1','CMSSW_7_1_0') )
            self.assertTrue( isNewerThan('CMSSW_5_3_1','CMSSW_5_3_1_patch1') )

            

    unittest.main()
