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
workflows[300]=['Pyquen_GammaJet_pt20_2760GeV',['Pyquen_GammaJet_pt20_2760GeV','DIGIHIMIX','RECOHIMIX','HARVESTHI2018']]
workflows[301]=['Pyquen_DiJet_pt80to120_2760GeV',['Pyquen_DiJet_pt80to120_2760GeV','DIGIHIMIX','RECOHIMIX','HARVESTHI2018']]
workflows[302]=['Pyquen_ZeemumuJets_pt10_2760GeV',['Pyquen_ZeemumuJets_pt10_2760GeV','DIGIHIMIX','RECOHIMIX','HARVESTHI2018']]

# 50 ns at 13 TeV and POSTLS1
workflows[50200]=['',['ZEE_13','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]
workflows[50202]=['',['TTbar_13','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]
workflows[50203]=['',['H125GGgluonfusion_13','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]
workflows[50204]=['',['QQH1352T_13','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]
workflows[50205]=['',['ZTT_13','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]
workflows[50206]=['',['ZMM_13','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]
workflows[50207]=['',['NuGun_UP15','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]
workflows[50208]=['',['SMS-T1tttt_mGl-1500_mLSP-100_13','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]

# 25 ns at 13 TeV and POSTLS1
workflows[25200]=['',['ZEE_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25202]=['',['TTbar_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25203]=['',['H125GGgluonfusion_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25204]=['',['QQH1352T_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25205]=['',['ZTT_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25206]=['',['ZMM_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25207]=['',['NuGun_UP15','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25208]=['',['SMS-T1tttt_mGl-1500_mLSP-100_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25209]=['',['QCD_FlatPt_15_3000HS_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25214]=['',['TTbarLepton_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]

# LHE-based fullSim PU  workflows
workflows[25210]=['',['TTbar012Jets_NLO_Mad_py8_Evt_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25211]=['',['GluGluHToZZTo4L_M125_Pow_py8_Evt_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25212]=['',['VBFHToZZTo4Nu_M125_Pow_py8_Evt_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]
workflows[25213]=['',['VBFHToBB_M125_Pow_py8_Evt_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]

# LHE-based fullSim PU workflows (2017)
workflows[25211.17]=['',['GluGluHToZZTo4L_M125_Pow_py8_Evt_13UP17','DIGIUP17_PU25','RECOUP17_PU25','HARVESTUP17_PU25']]
workflows[25212.17]=['',['VBFHToZZTo4Nu_M125_Pow_py8_Evt_13UP17','DIGIUP17_PU25','RECOUP17_PU25','HARVESTUP17_PU25']]
workflows[25213.17]=['',['VBFHToBB_M125_Pow_py8_Evt_13UP17','DIGIUP17_PU25','RECOUP17_PU25','HARVESTUP17_PU25']]

#fastsim
workflows[25400] = ['ZEE_13',["FS_ZEE_13_UP15_PU25","HARVESTUP15FS","MINIAODMCUP15FS"]]
workflows[25402] = ['TTbar_13',["FS_TTbar_13_UP15_PU25","HARVESTUP15FS","MINIAODMCUP15FS"]]
workflows[25403] = ['H125GGgluonfusion_13',["FS_H125GGgluonfusion_13_UP15_PU25","HARVESTUP15FS","MINIAODMCUP15FS"]]
#workflow[25404]
workflows[25405] = ['ZTT_13',["FS_ZTT_13_UP15_PU25","HARVESTUP15FS","MINIAODMCUP15FS"]]
workflows[25406] = ['ZMM_13',["FS_ZMM_13_UP15_PU25","HARVESTUP15FS","MINIAODMCUP15FS"]]
workflows[25407] = ['NuGun_UP15',["FS_NuGun_UP15_UP15_PU25","HARVESTUP15FS","MINIAODMCUP15FS"]]
workflows[25408] = ['SMS-T1tttt_mGl-1500_mLSP-100_13',["FS_SMS-T1tttt_mGl-1500_mLSP-100_13_UP15_PU25","HARVESTUP15FS","MINIAODMCUP15FS"]]
workflows[25409] = ['QCD_FlatPt_15_3000HS_13',["FS_QCD_FlatPt_15_3000HS_13_UP15_PU25","HARVESTUP15FS","MINIAODMCUP15FS"]] 

