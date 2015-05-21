# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

# 50 ns at 8 TeV
workflows[200]=['',['ZEE','DIGIPU1','RECOPU1','HARVEST']]
#workflows[201]=['',['ZmumuJets_Pt_20_300','DIGIPU2','RECOPU2','HARVEST']]
workflows[202]=['',['TTbar','DIGIPU1','RECOPU1','HARVEST']]
workflows[203]=['',['H130GGgluonfusion','DIGIPU1','RECOPU1','HARVEST']]
workflows[204]=['',['QQH1352T','DIGIPU1','RECOPU1','HARVEST']]
workflows[205]=['',['ZTT','DIGIPU1','RECOPU1','HARVEST']]

#heavy ions tests
workflows[300]=['Pyquen_GammaJet_pt20_2760GeV',['HydjetQ_MinBias_2760GeVINPUT','Pyquen_GammaJet_pt20_2760GeV','DIGIHISt3','RECOHISt4','HARVESTHI']]
workflows[301]=['Pyquen_DiJet_pt80to120_2760GeV',['HydjetQ_MinBias_2760GeVINPUT','Pyquen_DiJet_pt80to120_2760GeV','DIGIHISt3','RECOHISt4','HARVESTHI']]
workflows[302]=['Pyquen_ZeemumuJets_pt10_2760GeV',['HydjetQ_MinBias_2760GeVINPUT','Pyquen_ZeemumuJets_pt10_2760GeV','DIGIHISt3','RECOHISt4','HARVESTHI']]

#fastsim
workflows[400]=['TTbar',['TTbarFSPU','HARVESTFS']]
# temporarily redefine test 401 to pass tests during pu cfg transitation of FastSim
#workflows[401]=['TTbarNewMix',['TTbarFSPU2','HARVESTFS','MINIAODMCUP15FS']]
workflows[401]=['TTbarNewMix',['TTbarFSPU','HARVESTFS','MINIAODMCUP15FS']]


# 50 ns at 13 TeV and POSTLS1
workflows[50200]=['',['ZEE_13','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]
workflows[50202]=['',['TTbar_13','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]
workflows[50203]=['',['H130GGgluonfusion_13','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]
workflows[50204]=['',['QQH1352T_13','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]
workflows[50205]=['',['ZTT_13','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]
workflows[50206]=['',['ZMM_13','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]
workflows[50207]=['',['NuGun_UP15','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]
workflows[50208]=['',['SMS-T1tttt_mGl-1500_mLSP-100_13','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]

# 25 ns at 13 TeV and POSTLS1
workflows[25200]=['',['ZEE_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25202]=['',['TTbar_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25203]=['',['H130GGgluonfusion_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25204]=['',['QQH1352T_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25205]=['',['ZTT_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25206]=['',['ZMM_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25207]=['',['NuGun_UP15','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25208]=['',['SMS-T1tttt_mGl-1500_mLSP-100_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]

#fastsim
workflows[25400]=['',['FS_TTbar_13_PUAVE10','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[25400.1]=['',['FS_TTbar_13_PUAVE10_DRMIX_ITO','HARVESTUP15FS','MINIAODMCUP15FS']] #NEW MIX

workflows[25400.2]=['',['FS_TTbar_13_PUAVE35','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[25400.3]=['',['FS_TTbar_13_PUAVE35_DRMIX_ITO','HARVESTUP15FS','MINIAODMCUP15FS']] #NEW MIX

workflows[25401]=['',['FS_TTbar_13_PUAVE20','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[25402]=['',['FS_TTbar_13_PU25','HARVESTUP15FS','MINIAODMCUP15FS']] #NEW MIX
workflows[25407]=['',['FS_NuGun_UP15_PU25','HARVESTUP15FS','MINIAODMCUP15FS']] #NEW MIX
workflows[25408]=['',['FS_SMS-T1tttt_mGl-1500_mLSP-100_13_PU25','HARVESTUP15FS','MINIAODMCUP15FS']]
