import os
import re

def cmsswRelease():
    #return os.environ['CMSSW_BASE'].split('/')[-1]
    #this also works when the CMSSW directory is renamed
    return os.environ['CMSSW_VERSION']

def cmsswIs44X():
    return cmsswRelease().find('CMSSW_4_4_') != -1

def cmsswIs52X():
    if cmsswRelease().find('CMSSW_5_2_') != -1 or \
            cmsswRelease().find('CMSSW_5_3_') != -1:
        return True
    else:
        return False

def releaseNumber(release = None):
    # first check if this is an integration build
    if release is None:
        release = cmsswRelease()
    ibrel = re.compile('^CMSSW_(\d+)_(\d+)_X.*$')
    m = ibrel.match(release)
    if m:
        big = int(m.group(1))
        medium = int(m.group(2))
        return big, medium
    rerel = re.compile('^CMSSW_(\d+)_(\d+)_(\d+)(_\S+)*$')
    m = rerel.match(release)
    if m is None:
        raise ValueError('malformed release string '+release)
    big = int(m.group(1))
    medium = int(m.group(2))
    small = int(m.group(3))
    if m.group(4): # that's either a patch or prerelease
        prel = re.compile('_pre(\d+)')
        patch = re.compile('_patch(\d+)')
        pm = prel.match(m.group(4))
        if pm: # prerelease
            pre = int(pm.group(1))
            return big, medium, small, pre
        else: # patch
            pm2 = patch.match(m.group(4))
            if pm2:
                pat = int(pm2.group(1))
                return big, medium, small, pat
            else:
                raise ValueError('patch or prerelease string malformed '+m.group(4))
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
            out = releaseNumber('CMSSW_7_3_X_2014-10-30-0200')
            self.assertEqual(out, (7,3))
            out = releaseNumber('CMSSW_7_3_0_pre2')
            self.assertEqual(out, (7,3,0,2))
            self.assertRaises(ValueError, releaseNumber, 'foobar')
            self.assertRaises(ValueError, releaseNumber, 'CMSSW_1_2_3_xat3')
            self.assertRaises(ValueError, releaseNumber, 'CMSSW_1_2_a')
        def test_isNewerThan(self): 
            self.assertTrue( isNewerThan('CMSSW_5_3_1','CMSSW_7_1_0') )
            self.assertTrue( isNewerThan('CMSSW_5_3_1','CMSSW_5_3_1_patch1') )
            self.assertTrue( isNewerThan('CMSSW_5_3_1','CMSSW_5_3_1_pre1') )
            self.assertTrue( isNewerThan('CMSSW_5_3_1_pre1','CMSSW_5_3_1_pre2') )

            

    unittest.main()
