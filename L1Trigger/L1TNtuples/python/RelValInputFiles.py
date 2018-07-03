
# these files allow to test the workflows starting from different event content

# AOD
def RelValInputFile_AOD():
    return '/store/relval/CMSSW_7_5_0_pre1/RelValProdTTbar_13/AODSIM/MCRUN2_74_V7-v1/00000/48159643-5EE3-E411-818F-0025905A48F0.root'

# DIGI (only available in RelVal, not a realistic workflow)
def RelValInputFile_DIGI():
    return ''

# RAW (need to run RawToDigi to use this)
def RelValInputFile_RAW():
    return '/store/relval/CMSSW_7_5_0_pre4/RelValProdTTbar_13/GEN-SIM-RAW/MCRUN2_75_V1-v1/00000/1CFADAF5-E1F5-E411-A406-0025905A60D6.root'

#'/store/relval/CMSSW_7_5_0_pre1/RelValProdTTbar_13/GEN-SIM-RAW/MCRUN2_74_V7-v1/00000/0CEB1526-6CE3-E411-82B6-00261894386C.root'

