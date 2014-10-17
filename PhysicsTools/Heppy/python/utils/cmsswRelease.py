import os

def cmsswRelease():
    return os.environ['CMSSW_BASE'].split('/')[-1]

def cmsswIs44X():
    return cmsswRelease().find('CMSSW_4_4_') != -1
def cmsswIs52X():
    #return (cmsswRelease().find('CMSSW_5_2_') != -1 || cmsswRelease().find('CMSSW_5_3_') != -1)
    if cmsswRelease().find('CMSSW_5_2_') != -1:
        return 1
    if cmsswRelease().find('CMSSW_5_3_') != -1:
        return 1
    return 0

def parseReleaseString(release = None):
    
    if release is None:
        release = cmsswRelease()

    #split and remove 'CMSSW'
    tokens = release.split('_')[1:]
    
    output = []
    for t in tokens:
        try:
            output.append(int(t))
        except ValueError:
            output.append(t)
    if len(output) < 4:
        output.append('')
    return tuple(output[:4])

def isNewerThan(release1, release2=None):
    """Checks the orders of two releases. If release2 is not set, it is taken as the current release"""
    return parseReleaseString(release2) >= parseReleaseString(release1)


if __name__ == '__main__':
    print cmsswRelease()
    print 'is 44?', cmsswIs44X()
    print 'is 52?', cmsswIs52X()

    assert isNewerThan('CMSSW_4_4_4','CMSSW_5_3_0')
    assert isNewerThan('CMSSW_4_4_4_patch1','CMSSW_5_3_0')
    assert not isNewerThan('CMSSW_4_4_1','CMSSW_4_4_0')
    assert isNewerThan('CMSSW_4_9_9_patch1','CMSSW_5_0_0')
    assert not isNewerThan('CMSSW_4_4_4_patch2','CMSSW_4_4_4_patch1')
