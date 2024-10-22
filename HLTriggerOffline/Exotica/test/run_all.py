import os
#Create a list of samples to run, e.g.:
#samples = ["DisplacedSUSY","EtaBToJpsiJpsi","H125GGgluonfusion","MinBias","QCD_FlatPt_15_3000HS","QCD_Pt_1800_2400","SingleMuPt1000","SingleMuPt100","SingleMuPt10","TTbarToDilepton","TTbar","Upsilon1SToMuMu","WToLNu","WprimeToLNu","ZEE","ZMM","ZTT","ZpTT","ZpToEE"]

samples = []

for sample in samples:
  os.system("cmsRun hltExoticaValidator_cfg.py _input "+sample+"; cmsRun hltExoticaPostProcessor_cfg.py _input "+sample+"; cp DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root correctVal_DQM_"+sample+".root; root -l -b -q \"saveDQMHistograms.C(\\\"correctVal_DQM_"+sample+".root\\\",\\\"correct"+sample+"\\\")\";")
