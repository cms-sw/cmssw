# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

# FullSim, 8TeV, 50ns
workflows[200]=['',['ZEE','DIGIPU1','RECOPU1','HARVEST']]
workflows[201]=['',['ZmumuJets_Pt_20_300','DIGIPU1','RECOPU1','HARVEST']]
workflows[202]=['',['TTbar','DIGIPU1','RECOPU1','HARVEST']]
workflows[203]=['',['H130GGgluonfusion','DIGIPU1','RECOPU1','HARVEST']]
workflows[204]=['',['QQH1352T_Tauola','DIGIPU1','RECOPU1','HARVEST']]
workflows[205]=['',['ZTT','DIGIPU1','RECOPU1','HARVEST']]
workflows[206]=['',['ZMM','DIGIPU1','RECOPU1','HARVEST']]

# Heavy ions
workflows[300]=['Pyquen_GammaJet_pt20_2760GeV',['HydjetQ_MinBias_2760GeVINPUT','Pyquen_GammaJet_pt20_2760GeV','DIGIHISt3','RECOHISt4','HARVESTHI']]
workflows[301]=['Pyquen_DiJet_pt80to120_2760GeV',['HydjetQ_MinBias_2760GeVINPUT','Pyquen_DiJet_pt80to120_2760GeV','DIGIHISt3','RECOHISt4','HARVESTHI']]
workflows[302]=['Pyquen_ZeemumuJets_pt10_2760GeV',['HydjetQ_MinBias_2760GeVINPUT','Pyquen_ZeemumuJets_pt10_2760GeV','DIGIHISt3','RECOHISt4','HARVESTHI']]

# Fastsim, 8TeV, 50ns
workflows[400]=['TTbar',['TTbarFSPU','HARVESTFS']]
workflows[401]=['TTbarNewMix',['TTbarFSPU2','HARVESTFS']]

# FullSim, 13TeV, 50ns, POSTLS1
workflows[50200]=['',['ZEE_13','DIGIUP15_PU50','RECOUP15_PU50','HARVEST','MINIAODMC50']]
workflows[50201]=['',['ZmumuJets_Pt_20_300_13','DIGIUP15_PU50','RECOUP15_PU50','HARVEST','MINIAODMC50']]
workflows[50202]=['',['TTbar_13','DIGIUP15_PU50','RECOUP15_PU50','HARVEST','MINIAODMC','MINIAODMC50']]
workflows[50203]=['',['H130GGgluonfusion_13','DIGIUP15_PU50','RECOUP15_PU50','HARVEST','MINIAODMC50']]
workflows[50204]=['',['QQH1352T_Tauola_13','DIGIUP15_PU50','RECOUP15_PU50','HARVEST','MINIAODMC50']]
workflows[50205]=['',['ZTT_13','DIGIUP15_PU50','RECOUP15_PU50','HARVEST','MINIAODMC50']]
workflows[50206]=['',['ZMM_13','DIGIUP15_PU50','RECOUP15_PU50','HARVEST','MINIAODMC50']]

# FullSim, 13TeV, 25ns, POSTLS1
workflows[25200]=['',['ZEE_13','DIGIUP15_PU25','RECOUP15_PU25','HARVEST','MINIAODMC']]
workflows[25201]=['',['ZmumuJets_Pt_20_300_13','DIGIUP15_PU25','RECOUP15_PU25','HARVEST','MINIAODMC']]
workflows[25202]=['',['TTbar_13','DIGIUP15_PU25','RECOUP15_PU25','HARVEST','MINIAODMC']]
workflows[25203]=['',['H130GGgluonfusion_13','DIGIUP15_PU25','RECOUP15_PU25','HARVEST','MINIAODMC']]
workflows[25204]=['',['QQH1352T_Tauola_13','DIGIUP15_PU25','RECOUP15_PU25','HARVEST','MINIAODMC']]
workflows[25205]=['',['ZTT_13','DIGIUP15_PU25','RECOUP15_PU25','HARVEST','MINIAODMC']]
workflows[25206]=['',['ZMM_13','DIGIUP15_PU25','RECOUP15_PU25','HARVEST','MINIAODMC']]
