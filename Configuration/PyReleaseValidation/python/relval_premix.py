
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used


# premix at 13 TeV and POSTLS1
workflows[250199]=['',['PREMIXUP15_PU25']]
workflows[500199]=['',['PREMIXUP15_PU50']]

# 25ns pile up overlay using premix
workflows[250200]=['',['ZEE_13','DIGIPRMXUP15_PU25','RECOPRMXUP15_PU25','HARVESTUP15_PU25']]
workflows[250202]=['',['TTbar_13','DIGIPRMXUP15_PU25','RECOPRMXUP15_PU25','HARVESTUP15_PU25']]
workflows[250203]=['',['H130GGgluonfusion_13','DIGIPRMXUP15_PU25','RECOPRMXUP15_PU25','HARVESTUP15_PU25']]
workflows[250204]=['',['QQH1352T_13','DIGIPRMXUP15_PU25','RECOPRMXUP15_PU25','HARVESTUP15_PU25']]
workflows[250205]=['',['ZTT_13','DIGIPRMXUP15_PU25','RECOPRMXUP15_PU25','HARVESTUP15_PU25']]
workflows[250206]=['',['ZMM_13','DIGIPRMXUP15_PU25','RECOPRMXUP15_PU25','HARVESTUP15_PU25']]
workflows[250207]=['',['NuGun_UP15','DIGIPRMXUP15_PU25','RECOPRMXUP15_PU25','HARVESTUP15_PU25']]


# 50ns pile up overlay using premix
workflows[500200]=['',['ZEE_13','DIGIPRMXUP15_PU50','RECOPRMXUP15_PU50','HARVESTUP15_PU50']]
workflows[500202]=['',['TTbar_13','DIGIPRMXUP15_PU50','RECOPRMXUP15_PU50','HARVESTUP15_PU50']]
workflows[500203]=['',['H130GGgluonfusion_13','DIGIPRMXUP15_PU50','RECOPRMXUP15_PU50','HARVESTUP15_PU50']]
workflows[500204]=['',['QQH1352T_13','DIGIPRMXUP15_PU50','RECOPRMXUP15_PU50','HARVESTUP15_PU50']]
workflows[500205]=['',['ZTT_13','DIGIPRMXUP15_PU50','RECOPRMXUP15_PU50','HARVESTUP15_PU50']]
workflows[500206]=['',['ZMM_13','DIGIPRMXUP15_PU50','RECOPRMXUP15_PU50','HARVESTUP15_PU50']]
workflows[500207]=['',['NuGun_UP15','DIGIPRMXUP15_PU50','RECOPRMXUP15_PU50','HARVESTUP15_PU50']]

# develop pile up overlay using premix prod-like!
workflows[250200.1]=['ProdZEE_13_pmx25ns',['ProdZEE_13','DIGIPRMXUP15_PROD_PU25','RECOPRMXUP15PROD_PU25']]
workflows[500200.1]=['ProdZEE_13_pmx50ns',['ProdZEE_13','DIGIPRMXUP15_PROD_PU50','RECOPRMXUP15PROD_PU50']]

#fastsim, 25ns

## premixed minbias
workflows[250399]=['',['FS_PREMIXUP15_PU25']]
## signal + PU
workflows[250400] = ['ZEE_13',["FS_ZEE_13_PRMXUP15_PU25","HARVESTUP15FS","MINIAODMCUP15FS"]]
workflows[250402] = ['TTbar_13',["FS_TTbar_13_PRMXUP15_PU25","HARVESTUP15FS","MINIAODMCUP15FS"]]
workflows[250403] = ['H130GGgluonfusion_13',["FS_H130GGgluonfusion_13_PRMXUP15_PU25","HARVESTUP15FS","MINIAODMCUP15FS"]]
workflows[250405] = ['ZTT_13',["FS_ZTT_13_PRMXUP15_PU25","HARVESTUP15FS","MINIAODMCUP15FS"]]
workflows[250406] = ['ZMM_13',["FS_ZMM_13_PRMXUP15_PU25","HARVESTUP15FS","MINIAODMCUP15FS"]]
workflows[250407] = ['NUGUN_UP15',["FS_NuGun_UP15_PRMXUP15_PU25","HARVESTUP15FS","MINIAODMCUP15FS"]]
workflows[250408] = ['QCD_FlatPt_15_3000HS_13',["FS_QCD_FlatPt_15_3000HS_13_PRMXUP15_PU25","HARVESTUP15FS","MINIAODMCUP15FS"]]
workflows[250409] = ['SMS-T1tttt_mGl-1500_mLSP-100_13',["FS_SMS-T1tttt_mGl-1500_mLSP-100_13_PRMXUP15_PU25","HARVESTUP15FS","MINIAODMCUP15FS"]]
