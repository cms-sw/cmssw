from PhysicsTools.Heppy.utils.cmsswRelease import cmsswRelease, releaseNumber

def miniAodFiles():
    relnum = releaseNumber(cmsswRelease())
    files = []
    if relnum[:2]==(7,1):
        files = [
            '/store/relval/CMSSW_7_1_10_patch2/RelValZMM_13/MINIAODSIM/PU25ns_MCRUN2_71_V1-v1/00000/3E4EC015-AF53-E411-8889-0025905A6076.root',
                 '/store/relval/CMSSW_7_1_10_patch2/RelValZMM_13/MINIAODSIM/PU25ns_MCRUN2_71_V1-v1/00000/C483E714-AF53-E411-9B9A-0025905B855E.root'
            ]
    elif relnum[:2]==(7,0):
        files = [
            '/store/relval/CMSSW_7_0_9_patch3/RelValZMM_13/MINIAODSIM/PU25ns_PLS170_V7AN2-v1/00000/E0D7EDE4-0660-E411-B655-02163E00EF88.root',
             '/store/relval/CMSSW_7_0_9_patch3/RelValZMM_13/MINIAODSIM/PU25ns_PLS170_V7AN2-v1/00000/F0F88B6E-905F-E411-8080-0025904B0FBE.root'
            ]
    else:
        raise ValueError('no mini AOD file defined for release '+cmsswRelease())
    eosfiles = [''.join(['root://eoscms//eos/cms', lfn]) for lfn in files]
    return eosfiles


if __name__ == '__main__':
    print miniAodFiles()
