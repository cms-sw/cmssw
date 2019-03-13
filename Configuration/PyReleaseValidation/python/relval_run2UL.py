
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

# 2016 UL: 50ns pile up overlay using premix
#workflows[2500200.16]=['',['ZEE_13','DIGIPRMXUP15_NoHLT_PU50','ULHLT16','RECOPRMXUP15_PU50','HARVESTUP15_PU50']]

# 2017 UL: 25ns pile up overlay using premix
#Prod-link
workflows[2250200.17]=['',['ZEE_13UP17','DIGIPRMXUP17_NoHLT_PU25','ULHLT17','RECOPRMXUP17_PU25_PRODLIKE']]
workflows[2250202.17]=['',['TTbar_13UP17','DIGIPRMXUP17_NoHLT_PU25','ULHLT17','RECOPRMXUP17_PU25_PRODLIKE']]
workflows[2250203.17]=['',['H125GGgluonfusion_13UP17','DIGIPRMXUP17_NoHLT_PU25','ULHLT17','RECOPRMXUP17_PU25_PRODLIKE']]
workflows[2250204.17]=['',['QQH1352T_13UP17','DIGIPRMXUP17_NoHLT_PU25','ULHLT17','RECOPRMXUP17_PU25_PRODLIKE']]
workflows[2250205.17]=['',['ZTT_13UP17','DIGIPRMXUP17_NoHLT_PU25','ULHLT17','RECOPRMXUP17_PU25_PRODLIKE']]
workflows[2250206.17]=['',['ZMM_13UP17','DIGIPRMXUP17_NoHLT_PU25','ULHLT17','RECOPRMXUP17_PU25_PRODLIKE']]
workflows[2250207.17]=['',['NuGun_UP17','DIGIPRMXUP17_NoHLT_PU25','ULHLT17','RECOPRMXUP17_PU25_PRODLIKE']]
workflows[2250208.17]=['',['SMS-T1tttt_mGl-1500_mLSP-100_13UP17','DIGIPRMXUP17_NoHLT_PU25','ULHLT17','RECOPRMXUP17_PU25_PRODLIKE']]
#With validation
#workflows[2250200.17]=['',['ZEE_13UP17','DIGIPRMXUP17_NoHLT_PU25','ULHLT17','RECOPRMXUP17_PU25','HARVESTUP17_PU25']]  

#2018 UL: 25ns pile up overlay using premix
#workflows[2250200.18]=['',['ZEE_13UP18','DIGIPRMXUP18_NoHLT_PU25','ULHLT18','RECOPRMXUP18_PU25_L1TEgDQM','HARVESTUP18_PU25_L1TEgDQM']]
