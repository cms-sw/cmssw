from PhysicsTools.Heppy.utils.cmsswRelease import cmsswRelease, releaseNumber

def miniAodFiles():
    relnum = releaseNumber(cmsswRelease())
    files = []
    big, medium = relnum[:2] # e.g. 7, 3 for CMSSW_7_3_X
    if (big, medium)==(7,2):
        files = [
            '/store/relval/CMSSW_7_2_2_patch1/RelValZMM_13/MINIAODSIM/PU25ns_MCRUN2_72_V1-v1/00000/5C64D172-8D73-E411-9C77-0025905A60AA.root',
            '/store/relval/CMSSW_7_2_2_patch1/RelValZMM_13/MINIAODSIM/PU25ns_MCRUN2_72_V1-v1/00000/6CA2B9CC-8973-E411-8920-002618943904.root'
            ]
    elif (big,medium)==(7,1):
        files = [
            '/store/relval/CMSSW_7_1_10_patch2/RelValZMM_13/MINIAODSIM/PU25ns_MCRUN2_71_V1-v1/00000/3E4EC015-AF53-E411-8889-0025905A6076.root',
            '/store/relval/CMSSW_7_1_10_patch2/RelValZMM_13/MINIAODSIM/PU25ns_MCRUN2_71_V1-v1/00000/C483E714-AF53-E411-9B9A-0025905B855E.root'
            ]
    elif (big,medium)==(7,0):
        files = [
            '/store/relval/CMSSW_7_0_9_patch3/RelValZMM_13/MINIAODSIM/PU25ns_PLS170_V7AN2-v1/00000/E0D7EDE4-0660-E411-B655-02163E00EF88.root',
            '/store/relval/CMSSW_7_0_9_patch3/RelValZMM_13/MINIAODSIM/PU25ns_PLS170_V7AN2-v1/00000/F0F88B6E-905F-E411-8080-0025904B0FBE.root'
            ]
    elif (big,medium)==(7,3):
        files = [
            '/store/relval/CMSSW_7_3_0_pre1/RelValZMM_13/MINIAODSIM/PU25ns_PRE_LS172_V15-v1/00000/582D0582-355F-E411-9F30-02163E006D72.root'
            ]
    elif (big,medium)==(7,4):
        files = [
            '/store/relval/CMSSW_7_4_0_pre2/RelValZMM_13/MINIAODSIM/PU25ns_MCRUN2_73_V7-v1/00000/1ACF00C3-E09A-E411-BB34-0025905AA9F0.root',
            '/store/relval/CMSSW_7_4_0_pre2/RelValZMM_13/MINIAODSIM/PU25ns_MCRUN2_73_V7-v1/00000/4E1864C4-E09A-E411-98F8-0025905B861C.root'
            ]
    else:
        raise ValueError('no mini AOD file defined for release '+cmsswRelease())
    eosfiles = [''.join(['root://eoscms//eos/cms', lfn]) for lfn in files]
    return eosfiles


if __name__ == '__main__':
    print miniAodFiles()
