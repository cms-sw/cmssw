# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

# 25 ns at 13 TeV and POSTLS1
workflows[252001]=['',['ZEE_13MINIAOD','MINIAODMCINPUT']]
workflows[252011]=['',['ZmumuJets_Pt_20_300_13MINIAOD','MINIAODMCINPUT']]
workflows[252021]=['',['TTbar_13MINIAOD','MINIAODMCINPUT']]
workflows[252031]=['',['H130GGgluonfusion_13MINIAOD','MINIAODMCINPUT']]
workflows[252041]=['',['QQH1352T_Tauola_13MINIAOD','MINIAODMCINPUT']]
workflows[252051]=['',['ZTT_13MINIAOD','MINIAODMCINPUT']]

# real data
#workflows[4.711] = ['',['RunMinBias2012DMINIAOD','MINIAODDATAINPUT']]
#workflows[4.721] = ['',['RunMu2012DMINIAOD','MINIAODDATAINPUT']]
#workflows[4.731] = ['',['RunPhoton2012DMINIAOD','MINIAODDATAINPUT']]
#workflows[4.741] = ['',['RunEl2012DMINIAOD','MINIAODDATAINPUT']]
workflows[4.751] = ['',['RunJet2012DMINIAOD','MINIAODDATAINPUT']]
