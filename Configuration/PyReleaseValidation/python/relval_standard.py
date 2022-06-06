# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()


overridesEv5={'-n':'5'}

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

## production tests
workflows[1] = ['', ['ProdMinBias','DIGIPROD1','RECOPROD1']]
workflows[2] = ['', ['ProdTTbar','DIGIPROD1','RECOPROD1']]
workflows[3] = ['', ['ProdQCD_Pt_3000_3500','DIGIPROD1','RECOPROD1']]
workflows.addOverride(3,overridesEv5)                                                      #######################################################
###Prod2016
#workflows[1301] = ['', ['ProdMinBias_13','DIGIUP15PROD1','RECOPRODUP15','MINIAODMCUP15']] # Re-add miniaod here, to keep event content scrutinized
workflows[1301] = ['', ['ProdMinBias_13','DIGIUP15PROD1','RECOPRODUP15']]                  ######################################################## 
workflows[1302] = ['', ['ProdTTbar_13','DIGIUP15PROD1','RECOPRODUP15']]
workflows[1303] = ['', ['ProdQCD_Pt_3000_3500_13','DIGIUP15PROD1','RECOPRODUP15']]
workflows.addOverride(1303,overridesEv5)


#####Prod2017
workflows[1301.17] = ['', ['ProdMinBias_13UP17','DIGIUP17PROD1','RECOPRODUP17']]
workflows[1302.17] = ['', ['ProdTTbar_13UP17','DIGIUP17PROD1','RECOPRODUP17']]
workflows[1303.17] = ['', ['ProdQCD_Pt_3000_3500_13UP17','DIGIUP17PROD1','RECOPRODUP17']]
workflows.addOverride(1303.17,overridesEv5)

#####Prod2018
#workflows[1301.18] = ['', ['ProdMinBias_13UP18','DIGIUP18PROD1','RECOPRODUP18']]
workflows[1302.18] = ['', ['ProdTTbar_13UP18','DIGIUP18PROD1','RECOPRODUP18','MINIAODMCUP18','NANOPRODUP18']]
#workflows[1303.18] = ['', ['ProdQCD_Pt_3000_3500_13UP18','DIGIUP18PROD1','RECOPRODUP18']]
workflows[1304.18] = ['', ['ProdZEE_13UP18','DIGIUP18PROD1','RECOPRODUP18','MINIAODMCUP18','NANOPRODUP18']]
workflows[1304.182] = ['', ['ProdZEE_13UP18','DIGIUP18PROD1bParking','RECOPRODUP18bParking','MINIAODMCUP18bParking']]
#workflows.addOverride(1303.17,overridesEv5)

#####Prod2018 with concurrentlumi
workflows[1302.181] = ['', ['ProdTTbar_13UP18ml','DIGIUP18PROD1','RECOPRODUP18','MINIAODMCUP18','NANOPRODUP18']]
workflows[1304.181] = ['', ['ProdZEE_13UP18ml','DIGIUP18PROD1','RECOPRODUP18','MINIAODMCUP18','NANOPRODUP18']]

### data ###
workflows[4.5]  = ['', ['RunCosmicsA','RECOCOSD','ALCACOSD','HARVESTDC']]
#workflows[4.6]  = ['', ['MinimumBias2010A','RECODR1','HARVESTDR1']]
workflows[4.6]  = ['', ['MinimumBias2010A','RECOSKIMALCA','HARVESTDR1']]
#workflows[4.7]  = ['', ['MinimumBias2010B','RECODR1ALCA','HARVESTDR1']]
#workflows[4.8]  = ['', ['WZMuSkim2010A','RECODR1','HARVESTDR1']]
#workflows[4.9]  = ['', ['WZEGSkim2010A','RECODR1','HARVESTDR1']]
#workflows[4.10] = ['', ['WZMuSkim2010B','RECODR1','HARVESTDR1']]
#workflows[4.11] = ['', ['WZEGSkim2010B','RECODR1','HARVESTDR1']]

workflows[4.12] = ['', ['RunMinBias2010B','RECODR1','HARVESTDR1']]
#workflows[4.13] = ['', ['RunMu2010B','RECODR1','HARVESTDR1']]
#workflows[4.14] = ['', ['RunElectron2010B','RECODR1','HARVESTDR1']]
#workflows[4.15] = ['', ['RunPhoton2010B','RECODR1','HARVESTDR1']]
#workflows[4.16] = ['', ['RunJet2010B','RECODR1','HARVESTDR1']]


workflows[4.17] = ['', ['RunMinBias2011A','HLTD','RECODR1reHLT','HARVESTDR1reHLT','SKIMDreHLT']]
workflows[4.18] = ['', ['RunMu2011A','RECODR1','HARVESTDR1']]
workflows[4.19] = ['', ['RunElectron2011A','RECODR1','HARVESTDR1']]
workflows[4.20] = ['', ['RunPhoton2011A','RECODR1','HARVESTDR1']]
workflows[4.21] = ['', ['RunJet2011A','RECODR1','HARVESTDR1']]

workflows[4.22] = ['', ['RunCosmics2011A','RECOCOSD','ALCACOSD','SKIMCOSD','HARVESTDC']]
workflows[4.23] = ['',['ValSkim2011A','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.24] = ['',['WMuSkim2011A','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.25] = ['',['WElSkim2011A','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.26] = ['',['ZMuSkim2011A','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.27] = ['',['ZElSkim2011A','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.28] = ['',['HighMet2011A','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]

workflows[4.29] = ['', ['RunMinBias2011B','HLTD','RECODR1reHLT','HARVESTDR1reHLT','SKIMDreHLT']]
#workflows[4.291] = ['', ['RunMinBias2011B','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.30] = ['', ['RunMu2011B','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.31] = ['', ['RunElectron2011B','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.32] = ['', ['RunPhoton2011B','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.33] = ['', ['RunJet2011B','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]

workflows[4.34] = ['',['ValSkim2011B','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]
#workflows[4.35] = ['',['WMuSkim2011B','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.36] = ['',['WElSkim2011B','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.37] = ['',['ZMuSkim2011B','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.38] = ['',['ZElSkim2011B','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.39] = ['',['HighMet2011B','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]

workflows[4.40] = ['',['RunMinBias2012A','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.41] = ['',['RunTau2012A','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.42] = ['',['RunMET2012A','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.43] = ['',['RunMu2012A','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.44] = ['',['RunElectron2012A','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.45] = ['',['RunJet2012A','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]

workflows[4.51] = ['',['RunMinBias2012B','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.52] = ['',['RunMu2012B','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.53] = ['',['RunPhoton2012B','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.54] = ['',['RunEl2012B','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.55] = ['',['RunJet2012B','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.56] = ['',['ZMuSkim2012B','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.57] = ['',['ZElSkim2012B','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.58] = ['',['WElSkim2012B','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]

workflows[4.61] = ['',['RunMinBias2012C','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.62] = ['',['RunMu2012C','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.63] = ['',['RunPhoton2012C','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.64] = ['',['RunEl2012C','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.65] = ['',['RunJet2012C','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]

workflows[4.66] = ['',['ZMuSkim2012C','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.67] = ['',['ZElSkim2012C','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.68] = ['',['WElSkim2012C','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]

workflows[4.71] = ['',['RunMinBias2012D','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.72] = ['',['RunMu2012D','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.73] = ['',['RunPhoton2012D','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.74] = ['',['RunEl2012D','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.75] = ['',['RunJet2012D','HLTD','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.76] = ['',['ZMuSkim2012D','HLTDSKIM2','RECODR1reHLT2','HARVESTDR1reHLT']]
workflows[4.77] = ['',['ZElSkim2012D','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]
workflows[4.78] = ['',['WElSkim2012D','HLTDSKIM','RECODR1reHLT','HARVESTDR1reHLT']]

workflows[140.51] = ['',['RunHI2010','REPACKHID','RECOHID11St3','HARVESTDHI']]
workflows[140.52] = ['',['RunHI2010','RECOHID10','RECOHIR10D11','HARVESTDHI']]
workflows[140.53] = ['',['RunHI2011','RECOHID11','HARVESTDHI']]
workflows[140.54] = ['',['RunPA2013','RECO_PPbData','HARVEST_PPbData']]

workflows[140.55] = ['',['RunHI2015VR','HYBRIDRepackHI2015VR','HYBRIDZSHI2015','RECOHID15','HARVESTDHI15']]
workflows[140.56] = ['',['RunHI2018','RECOHID18','HARVESTDHI18']]
workflows[140.5611] = ['',['RunHI2018AOD','REMINIAODHID18','HARVESTHI18MINIAOD']]
workflows[140.57] = ['',['RunHI2018Reduced','RECOHID18','HARVESTDHI18']]

### run2 2015B 50ns ###
workflows[134.701] = ['',['RunHLTPhy2015B','HLTDR2_50ns','RECODR2_50nsreHLT_HIPM','HARVESTDR2']]
workflows[134.702] = ['',['RunDoubleEG2015B','HLTDR2_50ns','RECODR2_50nsreHLT_HIPM','HARVESTDR2']]
workflows[134.703] = ['',['RunDoubleMuon2015B','HLTDR2_50ns','RECODR2_50nsreHLT_HIPM','HARVESTDR2']]
workflows[134.704] = ['',['RunJetHT2015B','HLTDR2_50ns','RECODR2_50nsreHLT_HIPM','HARVESTDR2']]
workflows[134.705] = ['',['RunMET2015B','HLTDR2_50ns','RECODR2_50nsreHLT_HIPM','HARVESTDR2']]
workflows[134.706] = ['',['RunMuonEG2015B','HLTDR2_50ns','RECODR2_50nsreHLT_HIPM','HARVESTDR2']]
workflows[134.707] = ['',['RunSingleEl2015B','HLTDR2_50ns','RECODR2_50nsreHLT_HIPM','HARVESTDR2']]
workflows[134.708] = ['',['RunSingleMu2015B','HLTDR2_50ns','RECODR2_50nsreHLT_HIPM','HARVESTDR2']]
workflows[134.709] = ['',['RunSinglePh2015B','HLTDR2_50ns','RECODR2_50nsreHLT_HIPM','HARVESTDR2']]
workflows[134.710] = ['',['RunZeroBias2015B','HLTDR2_50ns','RECODR2_50nsreHLT_ZB_HIPM','HARVESTDR2ZB']]

### run 2015C 25ns ###
workflows[134.801] = ['',['RunHLTPhy2015C','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.802] = ['',['RunDoubleEG2015C','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.803] = ['',['RunDoubleMuon2015C','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.804] = ['',['RunJetHT2015C','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.805] = ['',['RunMET2015C','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.806] = ['',['RunMuonEG2015C','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.807] = ['',['RunDoubleEGPrpt2015C','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.808] = ['',['RunSingleMuPrpt2015C','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.809] = ['',['RunSingleEl2015C','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.810] = ['',['RunSingleMu2015C','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.811] = ['',['RunSinglePh2015C','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.812] = ['',['RunZeroBias2015C','HLTDR2_25ns','RECODR2_25nsreHLT_ZB_HIPM','HARVESTDR2ZB']]
workflows[134.813] = ['',['RunCosmics2015C','RECOCOSDRUN2','ALCACOSDRUN2','HARVESTDCRUN2']]

### run 2015D 25ns ###
workflows[134.901] = ['',['RunHLTPhy2015D','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.902] = ['',['RunDoubleEG2015D','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.903] = ['',['RunDoubleMuon2015D','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.904] = ['',['RunJetHT2015D','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.905] = ['',['RunMET2015D','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.906] = ['',['RunMuonEG2015D','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.907] = ['',['RunDoubleEGPrpt2015D','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.908] = ['',['RunSingleMuPrpt2015D','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.909] = ['',['RunSingleEl2015D','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.910] = ['',['RunSingleMu2015D','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.911] = ['',['RunSinglePh2015D','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
workflows[134.912] = ['',['RunZeroBias2015D','HLTDR2_25ns','RECODR2_25nsreHLT_ZB_HIPM','HARVESTDR2ZB']]

### run 2016B ###
workflows[136.721] = ['',['RunHLTPhy2016B','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.722] = ['',['RunDoubleEG2016B','HLTDR2_2016','RECODR2_2016reHLT_skimDoubleEG_HIPM','HARVESTDR2_skimDoubleEG']]
workflows[136.723] = ['',['RunDoubleMuon2016B','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.724] = ['',['RunJetHT2016B','HLTDR2_2016','RECODR2_2016reHLT_skimJetHT_HIPM','HARVESTDR2_skimJetHT']]
workflows[136.725] = ['',['RunMET2016B','HLTDR2_2016','RECODR2_2016reHLT_skimMET_HIPM','HARVESTDR2_skimMET']]
workflows[136.726] = ['',['RunMuonEG2016B','HLTDR2_2016','RECODR2_2016reHLT_skimMuonEG_HIPM','HARVESTDR2_skimMuonEG']]
workflows[136.727] = ['',['RunDoubleEGPrpt2016B','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.728] = ['',['RunSingleMuPrpt2016B','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.729] = ['',['RunSingleEl2016B','HLTDR2_2016','RECODR2_2016reHLT_L1TEgDQM_HIPM','HARVEST2016_L1TEgDQM']]
workflows[136.730] = ['',['RunSingleMu2016B','HLTDR2_2016','RECODR2_2016reHLT_skimSingleMu_HIPM','HARVEST2016_L1TMuDQM']]
workflows[136.731] = ['',['RunSinglePh2016B','HLTDR2_2016','RECODR2_2016reHLT_skimSinglePh_HIPM','HARVESTDR2_skimSinglePh']]
workflows[136.732] = ['',['RunZeroBias2016B','HLTDR2_2016','RECODR2_2016reHLT_ZB_HIPM','HARVESTDR2ZB']]
workflows[136.733] = ['',['RunCosmics2016B','RECOCOSDRUN2','ALCACOSDRUN2','HARVESTDCRUN2']]
workflows[136.734] = ['',['RunMuOnia2016B','HLTDR2_2016','RECODR2_2016reHLT_skimMuOnia_HIPM','HARVESTDR2_skimMuOnia']]
workflows[136.735] = ['',['RunNoBPTX2016B','HLTDR2_2016','RECODR2reHLTAlCaTkCosmics_HIPM','HARVESTDR2']]
# reminiAOD wf on 2016B input, mainly here for PPS testing
workflows[136.72411] = ['',['RunJetHT2016B_reminiaodUL','REMINIAOD_data2016UL_HIPM','HARVESTDR2_REMINIAOD_data2016UL_HIPM']]
workflows[136.72412] = ['',['RunJetHT2016B_reminiaodUL','REMININANO_data2016UL_HIPM','HARVESTDR2_REMININANO_data2016UL_HIPM']]
#workflows[136.72413] = ['',['RunJetHT2016B_reminiaodUL','REMININANO_data2016UL_HIPM_met','HARVESTDR2_REMININANO_data2016UL_HIPM_met']] #diable for now until MET is fixed.

### run 2016C ###
workflows[136.736] = ['',['RunHLTPhy2016C','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.737] = ['',['RunDoubleEG2016C','HLTDR2_2016','RECODR2_2016reHLT_skimDoubleEG_HIPM','HARVESTDR2_skimDoubleEG']]
workflows[136.738] = ['',['RunDoubleMuon2016C','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.739] = ['',['RunJetHT2016C','HLTDR2_2016','RECODR2_2016reHLT_skimJetHT_HIPM','HARVESTDR2_skimJetHT']]
workflows[136.740] = ['',['RunMET2016C','HLTDR2_2016','RECODR2_2016reHLT_skimMET_HIPM','HARVESTDR2_skimMET']]
workflows[136.741] = ['',['RunMuonEG2016C','HLTDR2_2016','RECODR2_2016reHLT_skimMuonEG_HIPM','HARVESTDR2_skimMuonEG']]
workflows[136.742] = ['',['RunSingleEl2016C','HLTDR2_2016','RECODR2_2016reHLT_L1TEgDQM_HIPM','HARVEST2016_L1TEgDQM']]
workflows[136.743] = ['',['RunSingleMu2016C','HLTDR2_2016','RECODR2_2016reHLT_L1TMuDQM_HIPM','HARVEST2016_L1TMuDQM']]
workflows[136.744] = ['',['RunSinglePh2016C','HLTDR2_2016','RECODR2_2016reHLT_skimSinglePh_HIPM','HARVESTDR2_skimSinglePh']]
workflows[136.745] = ['',['RunZeroBias2016C','HLTDR2_2016','RECODR2_2016reHLT_ZB_HIPM','HARVESTDR2ZB']]
workflows[136.746] = ['',['RunMuOnia2016C','HLTDR2_2016','RECODR2_2016reHLT_skimMuOnia_HIPM','HARVESTDR2_skimMuOnia']]

### run 2016D ###
workflows[136.747] = ['',['RunHLTPhy2016D','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.748] = ['',['RunDoubleEG2016D','HLTDR2_2016','RECODR2_2016reHLT_skimDoubleEG_HIPM','HARVESTDR2_skimDoubleEG']]
workflows[136.749] = ['',['RunDoubleMuon2016D','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.750] = ['',['RunJetHT2016D','HLTDR2_2016','RECODR2_2016reHLT_skimJetHT_HIPM','HARVESTDR2_skimJetHT']]
workflows[136.751] = ['',['RunMET2016D','HLTDR2_2016','RECODR2_2016reHLT_skimMET_HIPM','HARVESTDR2_skimMET']]
workflows[136.752] = ['',['RunMuonEG2016D','HLTDR2_2016','RECODR2_2016reHLT_skimMuonEG_HIPM','HARVESTDR2_skimMuonEG']]
workflows[136.753] = ['',['RunSingleEl2016D','HLTDR2_2016','RECODR2_2016reHLT_L1TEgDQM_HIPM','HARVEST2016_L1TEgDQM']]
workflows[136.754] = ['',['RunSingleMu2016D','HLTDR2_2016','RECODR2_2016reHLT_L1TMuDQM_HIPM','HARVEST2016_L1TMuDQM']]
workflows[136.755] = ['',['RunSinglePh2016D','HLTDR2_2016','RECODR2_2016reHLT_skimSinglePh_HIPM','HARVESTDR2_skimSinglePh']]
workflows[136.756] = ['',['RunZeroBias2016D','HLTDR2_2016','RECODR2_2016reHLT_ZB_HIPM','HARVESTDR2ZB']]
workflows[136.757] = ['',['RunMuOnia2016D','HLTDR2_2016','RECODR2_2016reHLT_skimMuOnia_HIPM','HARVESTDR2_skimMuOnia']]

### run 2016E ###
workflows[136.758] = ['',['RunHLTPhy2016E','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.759] = ['',['RunDoubleEG2016E','HLTDR2_2016','RECODR2_2016reHLT_skimDoubleEG_HIPM','HARVESTDR2_skimDoubleEG']]
workflows[136.760] = ['',['RunDoubleMuon2016E','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.761] = ['',['RunJetHT2016E','HLTDR2_2016','RECODR2_2016reHLT_skimJetHT_HIPM','HARVESTDR2_skimJetHT']]
workflows[136.762] = ['',['RunMET2016E','HLTDR2_2016','RECODR2_2016reHLT_skimMET_HIPM','HARVESTDR2_skimMET']]
workflows[136.763] = ['',['RunMuonEG2016E','HLTDR2_2016','RECODR2_2016reHLT_skimMuonEG_HIPM','HARVESTDR2_skimMuonEG']]
workflows[136.764] = ['',['RunSingleEl2016E','HLTDR2_2016','RECODR2_2016reHLT_L1TEgDQM_HIPM','HARVEST2016_L1TEgDQM']]
workflows[136.765] = ['',['RunSingleMu2016E','HLTDR2_2016','RECODR2_2016reHLT_L1TMuDQM_HIPM','HARVEST2016_L1TMuDQM']]
workflows[136.766] = ['',['RunSinglePh2016E','HLTDR2_2016','RECODR2_2016reHLT_skimSinglePh_HIPM','HARVESTDR2_skimSinglePh']]
workflows[136.767] = ['',['RunZeroBias2016E','HLTDR2_2016','RECODR2_2016reHLT_ZB_HIPM','HARVESTDR2ZB']]
workflows[136.768] = ['',['RunMuOnia2016E','HLTDR2_2016','RECODR2_2016reHLT_skimMuOnia_HIPM','HARVESTDR2_skimMuOnia']]
# reminiAOD wf on 2016E input
workflows[136.7611] = ['',['RunJetHT2016E_reminiaod','REMINIAOD_data2016_HIPM','HARVESTDR2_REMINIAOD_data2016_HIPM']]
workflows[136.76111] = ['',['RunJetHT2016E_reminiaodUL','REMINIAOD_data2016UL_HIPM','HARVESTDR2_REMINIAOD_data2016UL_HIPM']]

### run 2016H ###
workflows[136.769] = ['',['RunHLTPhy2016H','HLTDR2_2016','RECODR2_2016reHLT_Prompt','HARVESTDR2']]
workflows[136.770] = ['',['RunDoubleEG2016H','HLTDR2_2016','RECODR2_2016reHLT_skimDoubleEG_Prompt','HARVESTDR2_skimDoubleEG']]
workflows[136.771] = ['',['RunDoubleMuon2016H','HLTDR2_2016','RECODR2_2016reHLT_Prompt','HARVESTDR2']]
workflows[136.772] = ['',['RunJetHT2016H','HLTDR2_2016','RECODR2_2016reHLT_skimJetHT_Prompt','HARVESTDR2_skimJetHT']]
workflows[136.773] = ['',['RunMET2016H','HLTDR2_2016','RECODR2_2016reHLT_skimMET_Prompt','HARVESTDR2_skimMET']]
workflows[136.774] = ['',['RunMuonEG2016H','HLTDR2_2016','RECODR2_2016reHLT_skimMuonEG_Prompt','HARVESTDR2_skimMuonEG']]
workflows[136.775] = ['',['RunSingleEl2016H','HLTDR2_2016','RECODR2_2016reHLT_Prompt_L1TEgDQM','HARVEST2016_L1TEgDQM']]
workflows[136.776] = ['',['RunSingleMu2016H','HLTDR2_2016','RECODR2_2016reHLT_Prompt_Lumi_L1TMuDQM','HARVEST2016_L1TMuDQM']]
workflows[136.777] = ['',['RunSinglePh2016H','HLTDR2_2016','RECODR2_2016reHLT_skimSinglePh_Prompt','HARVESTDR2_skimSinglePh']]
workflows[136.778] = ['',['RunZeroBias2016H','HLTDR2_2016','RECODR2_2016reHLT_ZBPrompt','HARVESTDR2ZB']]
workflows[136.779] = ['',['RunMuOnia2016H','HLTDR2_2016','RECODR2_2016reHLT_skimMuOnia_Prompt','HARVESTDR2_skimMuOnia']]
# reminiAOD wf on 2016H input
workflows[136.7721] = ['',['RunJetHT2016H_reminiaod','REMINIAOD_data2016','HARVESTDR2_REMINIAOD_data2016']]
workflows[136.77211] = ['',['RunJetHT2016H_reminiaodUL','REMINIAOD_data2016UL','HARVESTDR2_REMINIAOD_data2016UL']]
# nanoAOD wf on 2016H 80X input
workflows[136.7722] = ['',['RunJetHT2016H_nano','NANOEDM2016_80X','HARVESTNANOAOD2016_80X']]

### run 2017B ###
workflows[136.780] = ['',['RunHLTPhy2017B','HLTDR2_2017','RECODR2_2017reHLT_Prompt','HARVEST2017']]
workflows[136.781] = ['',['RunDoubleEG2017B','HLTDR2_2017','RECODR2_2017reHLT_skimDoubleEG_Prompt','HARVEST2017_skimDoubleEG']]
workflows[136.782] = ['',['RunDoubleMuon2017B','HLTDR2_2017','RECODR2_2017reHLT_Prompt','HARVEST2017']]
workflows[136.783] = ['',['RunJetHT2017B','HLTDR2_2017','RECODR2_2017reHLT_skimJetHT_Prompt','HARVEST2017_skimJetHT']]
workflows[136.784] = ['',['RunMET2017B','HLTDR2_2017','RECODR2_2017reHLT_skimMET_Prompt','HARVEST2017_skimMET']]
workflows[136.785] = ['',['RunMuonEG2017B','HLTDR2_2017','RECODR2_2017reHLT_skimMuonEG_Prompt','HARVEST2017_skimMuonEG']]
workflows[136.786] = ['',['RunSingleEl2017B','HLTDR2_2017','RECODR2_2017reHLT_Prompt_L1TEgDQM','HARVEST2017_L1TEgDQM']]
workflows[136.787] = ['',['RunSingleMu2017B','HLTDR2_2017','RECODR2_2017reHLT_skimSingleMu_Prompt_Lumi','HARVEST2017_L1TMuDQM']]
workflows[136.788] = ['',['RunSinglePh2017B','HLTDR2_2017','RECODR2_2017reHLT_skimSinglePh_Prompt','HARVEST2017_skimSinglePh']]
workflows[136.789] = ['',['RunZeroBias2017B','HLTDR2_2017','RECODR2_2017reHLT_ZBPrompt','HARVEST2017ZB']]
workflows[136.790] = ['',['RunMuOnia2017B','HLTDR2_2017','RECODR2_2017reHLT_skimMuOnia_Prompt','HARVEST2017_skimMuOnia']]
workflows[136.791] = ['',['RunNoBPTX2017B','HLTDR2_2017','RECODR2_2017reHLTAlCaTkCosmics_Prompt','HARVEST2017']]
workflows[136.7801] = ['',['RunHLTPhy2017B_AOD','DQMHLTonAOD_2017','HARVESTDQMHLTonAOD_2017']]
workflows[136.7802] = ['',['RunHLTPhy2017B_AODextra','DQMHLTonAODextra_2017','HARVESTDQMHLTonAOD_2017']]
workflows[136.7803] = ['',['RunHLTPhy2017B_RAWAOD','DQMHLTonRAWAOD_2017','HARVESTDQMHLTonAOD_2017']]
workflows[136.844] = ['',['RunCharmonium2017B','HLTDR2_2017','RECODR2_2017reHLT_skimCharmonium_Prompt','HARVEST2017_skimCharmonium']]

### run 2017C ###
workflows[136.792] = ['',['RunHLTPhy2017C','HLTDR2_2017','RECODR2_2017reHLT_Prompt','HARVEST2017']]
workflows[136.793] = ['',['RunDoubleEG2017C','HLTDR2_2017','RECODR2_2017reHLT_skimDoubleEG_Prompt','HARVEST2017_skimDoubleEG']]
workflows[136.794] = ['',['RunDoubleMuon2017C','HLTDR2_2017','RECODR2_2017reHLT_Prompt','HARVEST2017']]
workflows[136.795] = ['',['RunJetHT2017C','HLTDR2_2017','RECODR2_2017reHLT_skimJetHT_Prompt','HARVEST2017_skimJetHT']]
workflows[136.796] = ['',['RunMET2017C','HLTDR2_2017','RECODR2_2017reHLT_skimMET_Prompt','HARVEST2017_skimMET']]
workflows[136.797] = ['',['RunMuonEG2017C','HLTDR2_2017','RECODR2_2017reHLT_skimMuonEG_Prompt','HARVEST2017_skimMuonEG']]
workflows[136.798] = ['',['RunSingleEl2017C','HLTDR2_2017','RECODR2_2017reHLT_Prompt_L1TEgDQM','HARVEST2017_L1TEgDQM']]
workflows[136.799] = ['',['RunSingleMu2017C','HLTDR2_2017','RECODR2_2017reHLT_skimSingleMu_Prompt_Lumi','HARVEST2017_L1TMuDQM']]
workflows[136.800] = ['',['RunSinglePh2017C','HLTDR2_2017','RECODR2_2017reHLT_skimSinglePh_Prompt','HARVEST2017_skimSinglePh']]
workflows[136.801] = ['',['RunZeroBias2017C','HLTDR2_2017','RECODR2_2017reHLT_ZBPrompt','HARVEST2017ZB']]
workflows[136.802] = ['',['RunMuOnia2017C','HLTDR2_2017','RECODR2_2017reHLT_skimMuOnia_Prompt','HARVEST2017_skimMuOnia']]
workflows[136.803] = ['',['RunNoBPTX2017C','HLTDR2_2017','RECODR2_2017reHLTAlCaTkCosmics_Prompt','HARVEST2017']]
workflows[136.840] = ['',['RunDisplacedJet2017C','HLTDR2_2017','RECODR2_2017reHLT_skimDisplacedJet_Prompt','HARVEST2017_skimDisplacedJet']]
workflows[136.845] = ['',['RunCharmonium2017C','HLTDR2_2017','RECODR2_2017reHLT_skimCharmonium_Prompt','HARVEST2017_skimCharmonium']]

### run 2017D ###
workflows[136.804] = ['',['RunHLTPhy2017D','HLTDR2_2017','RECODR2_2017reHLT_Prompt','HARVEST2017']]
workflows[136.805] = ['',['RunDoubleEG2017D','HLTDR2_2017','RECODR2_2017reHLT_skimDoubleEG_Prompt','HARVEST2017_skimDoubleEG']]
workflows[136.806] = ['',['RunDoubleMuon2017D','HLTDR2_2017','RECODR2_2017reHLT_Prompt','HARVEST2017']]
workflows[136.807] = ['',['RunJetHT2017D','HLTDR2_2017','RECODR2_2017reHLT_skimJetHT_Prompt','HARVEST2017_skimJetHT']]
workflows[136.808] = ['',['RunMET2017D','HLTDR2_2017','RECODR2_2017reHLT_skimMET_Prompt','HARVEST2017_skimMET']]
workflows[136.809] = ['',['RunMuonEG2017D','HLTDR2_2017','RECODR2_2017reHLT_skimMuonEG_Prompt','HARVEST2017_skimMuonEG']]
workflows[136.810] = ['',['RunSingleEl2017D','HLTDR2_2017','RECODR2_2017reHLT_Prompt_L1TEgDQM','HARVEST2017_L1TEgDQM']]
workflows[136.811] = ['',['RunSingleMu2017D','HLTDR2_2017','RECODR2_2017reHLT_skimSingleMu_Prompt_Lumi','HARVEST2017_L1TMuDQM']]
workflows[136.812] = ['',['RunSinglePh2017D','HLTDR2_2017','RECODR2_2017reHLT_skimSinglePh_Prompt','HARVEST2017_skimSinglePh']]
workflows[136.813] = ['',['RunZeroBias2017D','HLTDR2_2017','RECODR2_2017reHLT_ZBPrompt','HARVEST2017ZB']]
workflows[136.814] = ['',['RunMuOnia2017D','HLTDR2_2017','RECODR2_2017reHLT_skimMuOnia_Prompt','HARVEST2017_skimMuOnia']]
workflows[136.815] = ['',['RunNoBPTX2017D','HLTDR2_2017','RECODR2_2017reHLTAlCaTkCosmics_Prompt','HARVEST2017']]
workflows[136.841] = ['',['RunDisplacedJet2017D','HLTDR2_2017','RECODR2_2017reHLT_skimDisplacedJet_Prompt','HARVEST2017_skimDisplacedJet']]
workflows[136.846] = ['',['RunCharmonium2017D','HLTDR2_2017','RECODR2_2017reHLT_skimCharmonium_Prompt','HARVEST2017_skimCharmonium']]

### run 2017E ###
workflows[136.816] = ['',['RunHLTPhy2017E','HLTDR2_2017','RECODR2_2017reHLT_Prompt','HARVEST2017']]
workflows[136.817] = ['',['RunDoubleEG2017E','HLTDR2_2017','RECODR2_2017reHLT_skimDoubleEG_Prompt','HARVEST2017_skimDoubleEG']]
workflows[136.818] = ['',['RunDoubleMuon2017E','HLTDR2_2017','RECODR2_2017reHLT_Prompt','HARVEST2017']]
workflows[136.819] = ['',['RunJetHT2017E','HLTDR2_2017','RECODR2_2017reHLT_skimJetHT_Prompt','HARVEST2017_skimJetHT']]
workflows[136.820] = ['',['RunMET2017E','HLTDR2_2017','RECODR2_2017reHLT_skimMET_Prompt','HARVEST2017_skimMET']]
workflows[136.821] = ['',['RunMuonEG2017E','HLTDR2_2017','RECODR2_2017reHLT_skimMuonEG_Prompt','HARVEST2017_skimMuonEG']]
workflows[136.822] = ['',['RunSingleEl2017E','HLTDR2_2017','RECODR2_2017reHLT_Prompt_L1TEgDQM','HARVEST2017_L1TEgDQM']]
workflows[136.823] = ['',['RunSingleMu2017E','HLTDR2_2017','RECODR2_2017reHLT_skimSingleMu_Prompt_Lumi','HARVEST2017_L1TMuDQM']]
workflows[136.824] = ['',['RunSinglePh2017E','HLTDR2_2017','RECODR2_2017reHLT_skimSinglePh_Prompt','HARVEST2017_skimSinglePh']]
workflows[136.825] = ['',['RunZeroBias2017E','HLTDR2_2017','RECODR2_2017reHLT_ZBPrompt','HARVEST2017ZB']]
workflows[136.826] = ['',['RunMuOnia2017E','HLTDR2_2017','RECODR2_2017reHLT_skimMuOnia_Prompt','HARVEST2017_skimMuOnia']]
workflows[136.827] = ['',['RunNoBPTX2017E','HLTDR2_2017','RECODR2_2017reHLTAlCaTkCosmics_Prompt','HARVEST2017']]
workflows[136.842] = ['',['RunDisplacedJet2017E','HLTDR2_2017','RECODR2_2017reHLT_skimDisplacedJet_Prompt','HARVEST2017_skimDisplacedJet']]
workflows[136.847] = ['',['RunCharmonium2017E','HLTDR2_2017','RECODR2_2017reHLT_skimCharmonium_Prompt','HARVEST2017_skimCharmonium']]

### run 2017F ###
workflows[136.828] = ['',['RunHLTPhy2017F','HLTDR2_2017','RECODR2_2017reHLT_Prompt','HARVEST2017']]
workflows[136.829] = ['',['RunDoubleEG2017F','HLTDR2_2017','RECODR2_2017reHLT_skimDoubleEG_Prompt','HARVEST2017_skimDoubleEG']]
workflows[136.830] = ['',['RunDoubleMuon2017F','HLTDR2_2017','RECODR2_2017reHLT_Prompt','HARVEST2017']]
workflows[136.831] = ['',['RunJetHT2017F','HLTDR2_2017','RECODR2_2017reHLT_skimJetHT_Prompt','HARVEST2017_skimJetHT']]
workflows[136.832] = ['',['RunMET2017F','HLTDR2_2017','RECODR2_2017reHLT_skimMET_Prompt','HARVEST2017_skimMET']]
workflows[136.833] = ['',['RunMuonEG2017F','HLTDR2_2017','RECODR2_2017reHLT_skimMuonEG_Prompt','HARVEST2017_skimMuonEG']]
workflows[136.834] = ['',['RunSingleEl2017F','HLTDR2_2017','RECODR2_2017reHLT_Prompt_L1TEgDQM','HARVEST2017_L1TEgDQM']]
workflows[136.835] = ['',['RunSingleMu2017F','HLTDR2_2017','RECODR2_2017reHLT_skimSingleMu_Prompt_Lumi','HARVEST2017_L1TMuDQM']]
workflows[136.836] = ['',['RunSinglePh2017F','HLTDR2_2017','RECODR2_2017reHLT_skimSinglePh_Prompt','HARVEST2017_skimSinglePh']]
workflows[136.837] = ['',['RunZeroBias2017F','HLTDR2_2017','RECODR2_2017reHLT_ZBPrompt','HARVEST2017ZB']]
workflows[136.838] = ['',['RunMuOnia2017F','HLTDR2_2017','RECODR2_2017reHLT_skimMuOnia_Prompt','HARVEST2017_skimMuOnia']]
workflows[136.839] = ['',['RunNoBPTX2017F','HLTDR2_2017','RECODR2_2017reHLTAlCaTkCosmics_Prompt','HARVEST2017']]
workflows[136.8391] = ['',['RunExpressPhy2017F','HLTDR2_2017','RECODR2_2017reHLTSiPixelCalZeroBias_Prompt','HARVEST2017']]
workflows[136.843] = ['',['RunDisplacedJet2017F','HLTDR2_2017','RECODR2_2017reHLT_skimDisplacedJet_Prompt','HARVEST2017_skimDisplacedJet']]
workflows[136.848] = ['',['RunCharmonium2017F','HLTDR2_2017','RECODR2_2017reHLT_skimCharmonium_Prompt','HARVEST2017_skimCharmonium']]
# reminiAOD wf on 2017F input
workflows[136.8311] = ['',['RunJetHT2017F_reminiaod','REMINIAOD_data2017','HARVEST2017_REMINIAOD_data2017']]
workflows[136.83111] = ['',['RunJetHT2017F_reminiaodUL','REMINIAOD_data2017UL','HARVEST2017_REMINIAOD_data2017UL']]

# NANOAOD wf on 2017C miniAODv2 94X input
workflows[136.7952] = ['',['RunJetHT2017C_94Xv2NanoAODINPUT','NANOEDM2017_94XMiniAODv2','HARVESTNANOAOD2017_94XMiniAODv2']]

### run 2018A ###
workflows[136.849] = ['',['RunHLTPhy2018A','HLTDR2_2018','RECODR2_2018reHLT_Offline','HARVEST2018']]
workflows[136.850] = ['',['RunEGamma2018A','HLTDR2_2018','RECODR2_2018reHLT_skimEGamma_Offline_L1TEgDQM','HARVEST2018_L1TEgDQM']]
workflows[136.851] = ['',['RunDoubleMuon2018A','HLTDR2_2018','RECODR2_2018reHLT_Offline','HARVEST2018']]
workflows[136.852] = ['',['RunJetHT2018A','HLTDR2_2018','RECODR2_2018reHLT_skimJetHT_Offline','HARVEST2018_skimJetHT']]
workflows[136.853] = ['',['RunMET2018A','HLTDR2_2018','RECODR2_2018reHLT_skimMET_Offline','HARVEST2018_skimMET']]
workflows[136.854] = ['',['RunMuonEG2018A','HLTDR2_2018','RECODR2_2018reHLT_skimMuonEG_Offline','HARVEST2018_skimMuonEG']]
workflows[136.855] = ['',['RunSingleMu2018A','HLTDR2_2018','RECODR2_2018reHLT_skimSingleMu_Offline_Lumi','HARVEST2018_L1TMuDQM']]
workflows[136.856] = ['',['RunZeroBias2018A','HLTDR2_2018','RECODR2_2018reHLT_ZBOffline','HARVEST2018ZB']]
workflows[136.857] = ['',['RunMuOnia2018A','HLTDR2_2018','RECODR2_2018reHLT_skimMuOnia_Offline','HARVEST2018_skimMuOnia']]
workflows[136.858] = ['',['RunNoBPTX2018A','HLTDR2_2018','RECODR2_2018reHLTAlCaTkCosmics_Offline','HARVEST2018']]
workflows[136.859] = ['',['RunDisplacedJet2018A','HLTDR2_2018','RECODR2_2018reHLT_skimDisplacedJet_Offline','HARVEST2018_skimDisplacedJet']]
workflows[136.860] = ['',['RunCharmonium2018A','HLTDR2_2018','RECODR2_2018reHLT_skimCharmonium_Offline','HARVEST2018_skimCharmonium']]
### wf to test 90 m beta* Totem run reconstruction ###
workflows[136.8501] = ['',['RunEGamma2018A','HLTDR2_2018','RECODR2_2018reHLT_skimParkingBPH_Offline','HARVEST2018_skimParkingBPH']]
workflows[136.8561] = ['',['RunZeroBias_hBStarTk','HLTDR2_2018_hBStar','RECODR2_2018reHLT_Offline_hBStar','HARVEST2018_hBStar']]
workflows[136.8562] = ['',['RunZeroBias1_hBStarRP','HLTDR2_2018_hBStar','RECODR2_2018reHLT_Offline_hBStar','HARVEST2018_hBStar']]

### NANOAOD wf on 2018 prompt reco MINIADD
workflows[136.8521] = ['',['RunJetHT2018A_nano','NANOEDM2018_102Xv1','HARVESTNANOAOD2018_102Xv1']]
workflows[136.8522] = ['',['RunJetHT2018A_nanoUL','NANOEDM2018_106Xv1','HARVESTNANOAOD2018_106Xv1']]
workflows[136.8523] = ['',['RunJetHT2018C_nanoULremini','NANOEDM2018_106Xv2','HARVESTNANOAOD2018_106Xv2']]

### run 2018B ###
workflows[136.861] = ['',['RunHLTPhy2018B','HLTDR2_2018','RECODR2_2018reHLT_Offline','HARVEST2018']]
workflows[136.862] = ['',['RunEGamma2018B','HLTDR2_2018','RECODR2_2018reHLT_skimEGamma_Offline_L1TEgDQM','HARVEST2018_L1TEgDQM']]
workflows[136.863] = ['',['RunDoubleMuon2018B','HLTDR2_2018','RECODR2_2018reHLT_Offline','HARVEST2018']]
workflows[136.864] = ['',['RunJetHT2018B','HLTDR2_2018','RECODR2_2018reHLT_skimJetHT_Offline','HARVEST2018_skimJetHT']]
workflows[136.865] = ['',['RunMET2018B','HLTDR2_2018','RECODR2_2018reHLT_skimMET_Offline','HARVEST2018_skimMET']]
workflows[136.866] = ['',['RunMuonEG2018B','HLTDR2_2018','RECODR2_2018reHLT_skimMuonEG_Offline','HARVEST2018_skimMuonEG']]
workflows[136.867] = ['',['RunSingleMu2018B','HLTDR2_2018','RECODR2_2018reHLT_skimSingleMu_Offline_Lumi','HARVEST2018_L1TMuDQM']]
workflows[136.868] = ['',['RunZeroBias2018B','HLTDR2_2018','RECODR2_2018reHLT_ZBOffline','HARVEST2018ZB']]
workflows[136.869] = ['',['RunMuOnia2018B','HLTDR2_2018','RECODR2_2018reHLT_skimMuOnia_Offline','HARVEST2018_skimMuOnia']]
workflows[136.870] = ['',['RunNoBPTX2018B','HLTDR2_2018','RECODR2_2018reHLTAlCaTkCosmics_Offline','HARVEST2018']]
workflows[136.871] = ['',['RunDisplacedJet2018B','HLTDR2_2018','RECODR2_2018reHLT_skimDisplacedJet_Offline','HARVEST2018_skimDisplacedJet']]
workflows[136.872] = ['',['RunCharmonium2018B','HLTDR2_2018','RECODR2_2018reHLT_skimCharmonium_Offline','HARVEST2018_skimCharmonium']]
workflows[136.898] = ['',['RunParkingBPH2018B','HLTDR2_2018','RECODR2_2018reHLT_skimParkingBPH_Offline','HARVEST2018_skimParkingBPH']]

workflows[136.8642] = ['',['RunJetHT2018BHEfail','HLTDR2_2018','RECODR2_2018reHLT_skimJetHT_Prompt_HEfail','HARVEST2018_HEfail_skimJetHT']]

### run 2018C ###
workflows[136.873] = ['',['RunHLTPhy2018C','HLTDR2_2018','RECODR2_2018reHLT_Offline','HARVEST2018']]
workflows[136.874] = ['',['RunEGamma2018C','HLTDR2_2018','RECODR2_2018reHLT_skimEGamma_Offline_L1TEgDQM','HARVEST2018_L1TEgDQM']]
workflows[136.875] = ['',['RunDoubleMuon2018C','HLTDR2_2018','RECODR2_2018reHLT_Offline','HARVEST2018']]
workflows[136.876] = ['',['RunJetHT2018C','HLTDR2_2018','RECODR2_2018reHLT_skimJetHT_Offline','HARVEST2018_skimJetHT']]
workflows[136.877] = ['',['RunMET2018C','HLTDR2_2018','RECODR2_2018reHLT_skimMET_Offline','HARVEST2018_skimMET']]
workflows[136.878] = ['',['RunMuonEG2018C','HLTDR2_2018','RECODR2_2018reHLT_skimMuonEG_Offline','HARVEST2018_skimMuonEG']]
workflows[136.879] = ['',['RunSingleMu2018C','HLTDR2_2018','RECODR2_2018reHLT_skimSingleMu_Offline_Lumi','HARVEST2018_L1TMuDQM']]
workflows[136.880] = ['',['RunZeroBias2018C','HLTDR2_2018','RECODR2_2018reHLT_ZBOffline','HARVEST2018ZB']]
workflows[136.881] = ['',['RunMuOnia2018C','HLTDR2_2018','RECODR2_2018reHLT_skimMuOnia_Offline','HARVEST2018_skimMuOnia']]
workflows[136.882] = ['',['RunNoBPTX2018C','HLTDR2_2018','RECODR2_2018reHLTAlCaTkCosmics_Offline','HARVEST2018']]
workflows[136.883] = ['',['RunDisplacedJet2018C','HLTDR2_2018','RECODR2_2018reHLT_skimDisplacedJet_Offline','HARVEST2018_skimDisplacedJet']]
workflows[136.884] = ['',['RunCharmonium2018C','HLTDR2_2018','RECODR2_2018reHLT_skimCharmonium_Offline','HARVEST2018_skimCharmonium']]

### run 2018D ###
workflows[136.885] = ['',['RunHLTPhy2018D','HLTDR2_2018','RECODR2_2018reHLT_Prompt','HARVEST2018_Prompt']]
workflows[136.886] = ['',['RunEGamma2018D','HLTDR2_2018','RECODR2_2018reHLT_skimEGamma_Prompt_L1TEgDQM','HARVEST2018_L1TEgDQM_Prompt']]
workflows[136.887] = ['',['RunDoubleMuon2018D','HLTDR2_2018','RECODR2_2018reHLT_Prompt','HARVEST2018_Prompt']]
workflows[136.888] = ['',['RunJetHT2018D','HLTDR2_2018','RECODR2_2018reHLT_skimJetHT_Prompt','HARVEST2018_skimJetHT_Prompt']]
workflows[136.889] = ['',['RunMET2018D','HLTDR2_2018','RECODR2_2018reHLT_skimMET_Prompt','HARVEST2018_skimMET_Prompt']]
workflows[136.890] = ['',['RunMuonEG2018D','HLTDR2_2018','RECODR2_2018reHLT_skimMuonEG_Prompt','HARVEST2018_skimMuonEG_Prompt']]
workflows[136.891] = ['',['RunSingleMu2018D','HLTDR2_2018','RECODR2_2018reHLT_skimSingleMu_Prompt_Lumi','HARVEST2018_L1TMuDQM_Prompt']]
workflows[136.892] = ['',['RunZeroBias2018D','HLTDR2_2018','RECODR2_2018reHLT_ZBPrompt','HARVEST2018_PromptZB']]
workflows[136.893] = ['',['RunMuOnia2018D','HLTDR2_2018','RECODR2_2018reHLT_skimMuOnia_Prompt','HARVEST2018_skimMuOnia_Prompt']]
workflows[136.894] = ['',['RunNoBPTX2018D','HLTDR2_2018','RECODR2_2018reHLTAlCaTkCosmics_Prompt','HARVEST2018_Prompt']]
workflows[136.895] = ['',['RunDisplacedJet2018D','HLTDR2_2018','RECODR2_2018reHLT_skimDisplacedJet_Prompt','HARVEST2018_skimDisplacedJet_Prompt']]
workflows[136.896] = ['',['RunCharmonium2018D','HLTDR2_2018','RECODR2_2018reHLT_skimCharmonium_Prompt','HARVEST2018_skimCharmonium_Prompt']]
# reminiAOD wf on 2018D input
workflows[136.88811] = ['',['RunJetHT2018D_reminiaodUL','REMINIAOD_data2018UL','HARVEST2018_REMINIAOD_data2018UL']]

### run 2018D pixel tracks ###
workflows[136.8855] = ['',['RunHLTPhy2018D','HLTDR2_2018','RECODR2_2018reHLT_Prompt_pixelTrackingOnly','HARVEST2018_pixelTrackingOnly']]
workflows[136.885501] = ['',['RunHLTPhy2018D','HLTDR2_2018','RECODR2_2018reHLT_Patatrack_PixelOnlyCPU','HARVEST2018_pixelTrackingOnly']]
workflows[136.8885] = ['',['RunJetHT2018D','HLTDR2_2018','RECODR2_2018reHLT_Prompt_pixelTrackingOnly','HARVEST2018_pixelTrackingOnly']]
workflows[136.888501] = ['',['RunJetHT2018D','HLTDR2_2018','RECODR2_2018reHLT_Patatrack_PixelOnlyCPU','HARVEST2018_pixelTrackingOnly']]

### run 2018D ECAL-only ###
workflows[136.885511] = ['',['RunHLTPhy2018D','HLTDR2_2018','RECODR2_2018reHLT_ECALOnlyCPU','HARVEST2018_ECALOnly']]
workflows[136.888511] = ['',['RunJetHT2018D','HLTDR2_2018','RECODR2_2018reHLT_ECALOnlyCPU','HARVEST2018_ECALOnly']]

### run 2018D HCAL-only ###
workflows[136.885521] = ['',['RunHLTPhy2018D','HLTDR2_2018','RECODR2_2018reHLT_HCALOnlyCPU','HARVEST2018_HCALOnly']]
workflows[136.888521] = ['',['RunJetHT2018D','HLTDR2_2018','RECODR2_2018reHLT_HCALOnlyCPU','HARVEST2018_HCALOnly']]

### run 2021 CRUZET/CRAFT ###
workflows[136.897] = ['',['RunCosmics2021CRUZET','RECOCOSDRUN3','ALCACOSDRUN3','HARVESTDCR3']]
workflows[136.899] = ['',['RunCosmics2021CRAFT','RECOCOSDRUN3','ALCACOSDRUN3','HARVESTDCR3']]

# multi-run harvesting
workflows[137.8] = ['',['RunEGamma2018C','HLTDR2_2018','RECODR2_2018reHLT_skimEGamma_Offline_L1TEgDQM',
                        'RunEGamma2018D','HLTDR2_2018','RECODR2_2018reHLT_skimEGamma_Prompt_L1TEgDQM','HARVEST2018_L1TEgDQM_MULTIRUN']]
### Prompt / Express / Splashes ###
workflows[138.1] = ['PromptCosmics',['RunCosmics2021','RECOCOSDPROMPTRUN3','ALCACOSDPROMPTRUN3','HARVESTDCPROMPTRUN3']]
workflows[138.2] = ['ExpressCosmics',['RunCosmics2021','RECOCOSDEXPRUN3','ALCACOSDEXPRUN3','HARVESTDCEXPRUN3']]
workflows[138.3] = ['Splashes',['RunMinimumBias2021Splash','COPYPASTER3','RECODR3Splash','HARVESTDR3']]
workflows[138.4] = ['PromptCollisions',['RunMinimumBias2021','ALCARECOPROMPTR3','HARVESTDPROMPTR3']]
workflows[138.5] = ['ExpressCollisions',['RunMinimumBias2021','TIER0EXPRUN3','ALCARECOEXPR3','HARVESTDEXPR3']]

#### Test of lumi section boundary crossing with run2 2018D ####
workflows[136.8861] = ['',['RunEGamma2018Dml1','HLTDR2_2018ml','RECODR2_2018reHLT_skimEGamma_Prompt_L1TEgDQM','HARVEST2018_L1TEgDQM_Prompt']]
workflows[136.8862] = ['',['RunEGamma2018Dml2','HLTDR2_2018ml','RECODR2_2018reHLT_skimEGamma_Prompt_L1TEgDQM','HARVEST2018_L1TEgDQM_Prompt']]

#### Test of tau embed
workflows[136.9] = ['', ['RunDoubleMuon2016C', 'RAWRECOTE16', 'RAWRECOLHECLEANTE16', 'EMBEDHADTE16', 'EMBEDMINIAOD16']]

### run 2021 collisions ###
workflows[139.001] = ['',['RunMinimumBias2021','HLTDR3_2021','RECODR3_MinBiasOffline','HARVESTD2021MB']]
workflows[139.002] = ['',['RunZeroBias2021','HLTDR3_2021','RECODR3_ZBOffline','HARVESTD2021ZB']]
workflows[139.003] = ['',['RunHLTPhy2021','HLTDR3_2021','RECODR3_HLTPhysics_Offline','HARVESTD2021HLTPhy']]
workflows[139.004] = ['',['RunNoBPTX2021','HLTDR3_2021','RECODR3_AlCaTkCosmics_Offline','HARVESTDR3']]
workflows[139.005] = ['',['AlCaPhiSym2021','RECOALCAECALPHISYMDR3','ALCAECALPHISYM']]

### fastsim ###
workflows[5.1] = ['TTbar', ['TTbarFS','HARVESTFS']]
workflows[5.2] = ['SingleMuPt10', ['SingleMuPt10FS','HARVESTFS']]
workflows[5.3] = ['SingleMuPt100', ['SingleMuPt100FS','HARVESTFS']]
workflows[5.4] = ['ZEE', ['ZEEFS','HARVESTFS']]
workflows[5.5] = ['ZTT',['ZTTFS','HARVESTFS']]

workflows[5.6]  = ['QCD_FlatPt_15_3000', ['QCDFlatPt153000FS','HARVESTFS']]
workflows[5.7] = ['H130GGgluonfusion', ['H130GGgluonfusionFS','HARVESTFS']]

### fastsim_13 TeV ###
workflows[135.1] = ['TTbar_13', ['TTbarFS_13','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.2] = ['SingleMuPt10_UP15', ['SingleMuPt10FS_UP15','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.3] = ['SingleMuPt100_UP15', ['SingleMuPt100FS_UP15','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.4] = ['ZEE_13', ['ZEEFS_13','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.5] = ['ZTT_13',['ZTTFS_13','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.6] = ['QCD_FlatPt_15_3000_13', ['QCDFlatPt153000FS_13','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.7] = ['H125GGgluonfusion_13', ['H125GGgluonfusionFS_13','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.9] = ['ZMM_13',['ZMMFS_13','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.11] = ['SMS-T1tttt_mGl-1500_mLSP-100_13', ['SMS-T1tttt_mGl-1500_mLSP-100FS_13','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.12] = ['QCD_Pt_80_120_13', ['QCD_Pt_80_120FS_13','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.13] = ['TTbar_13', ['TTbarFS_13_trackingOnlyValidation','HARVESTUP15FS_trackingOnly']]

### MinBias fastsim_13 TeV for mixing ###
workflows[135.8] = ['',['MinBiasFS_13_ForMixing']]

### Phase1 FastSim 13TeV ###                                                                                                                                                   
workflows[2017.1] = ['TTbar_13_UP17', ['TTbarFS_13_UP17','HARVESTUP17FS','MINIAODMCUP17FS']]
workflows[2017.2] = ['SingleMuPt10_UP17', ['SingleMuPt10FS_UP17','HARVESTUP17FS','MINIAODMCUP17FS']]
workflows[2017.3] = ['SingleMuPt100_UP17', ['SingleMuPt100FS_UP17','HARVESTUP17FS','MINIAODMCUP17FS']]
workflows[2017.4] = ['ZEE_13_UP17', ['ZEEFS_13_UP17','HARVESTUP17FS','MINIAODMCUP17FS']]
workflows[2017.5] = ['ZTT_13_UP17',['ZTTFS_13_UP17','HARVESTUP17FS','MINIAODMCUP17FS']]
workflows[2017.6] = ['QCD_FlatPt_15_3000_13_UP17', ['QCDFlatPt153000FS_13_UP17','HARVESTUP17FS','MINIAODMCUP17FS']]
workflows[2017.7] = ['H125GGgluonfusion_13_UP17', ['H125GGgluonfusionFS_13_UP17','HARVESTUP17FS','MINIAODMCUP17FS']]
workflows[2017.9] = ['ZMM_13_UP17',['ZMMFS_13_UP17','HARVESTUP17FS','MINIAODMCUP17FS']]
workflows[2017.11] = ['SMS-T1tttt_mGl-1500_mLSP-100_13_UP17', ['SMS-T1tttt_mGl-1500_mLSP-100FS_13_UP17','HARVESTUP17FS','MINIAODMCUP17FS']]
workflows[2017.12] = ['QCD_Pt_80_120_13_UP17', ['QCD_Pt_80_120FS_13_UP17','HARVESTUP17FS','MINIAODMCUP17FS']]
workflows[2017.13] = ['TTbar_13_UP17', ['TTbarFS_13_trackingOnlyValidation_UP17','HARVESTUP17FS_trackingOnly']]

### MinBias fastsim_13 TeV for mixing ###                                                                                                                                      
workflows[2017.8] = ['',['MinBiasFS_13_UP17_ForMixing']]

### MinBias fastsim_13 TeV for mixing, 2018 ###                                                                                                                                      
workflows[2018.8] = ['',['MinBiasFS_13_UP18_ForMixing']]

### Phase1 FastSim 13TeV, 2018 ###
workflows[2018.1] = ['TTbar_13_UP18', ['TTbarFS_13_UP18','HARVESTUP18FS','MINIAODMCUP18FS']]
workflows[2018.2] = ['SingleMuPt10_UP18', ['SingleMuPt10FS_UP18','HARVESTUP18FS','MINIAODMCUP18FS']]
workflows[2018.3] = ['SingleMuPt100_UP18', ['SingleMuPt100FS_UP18','HARVESTUP18FS','MINIAODMCUP18FS']]
workflows[2018.4] = ['ZEE_13_UP18', ['ZEEFS_13_UP18','HARVESTUP18FS','MINIAODMCUP18FS']]
workflows[2018.5] = ['ZTT_13_UP18',['ZTTFS_13_UP18','HARVESTUP18FS','MINIAODMCUP18FS']]
workflows[2018.6] = ['QCD_FlatPt_15_3000_13_UP18', ['QCDFlatPt153000FS_13_UP18','HARVESTUP18FS','MINIAODMCUP18FS']]
workflows[2018.7] = ['H125GGgluonfusion_13_UP18', ['H125GGgluonfusionFS_13_UP18','HARVESTUP18FS','MINIAODMCUP18FS']]
workflows[2018.9] = ['ZMM_13_UP18',['ZMMFS_13_UP18','HARVESTUP18FS','MINIAODMCUP18FS']]
workflows[2018.11] = ['SMS-T1tttt_mGl-1500_mLSP-100_13_UP18', ['SMS-T1tttt_mGl-1500_mLSP-100FS_13_UP18','HARVESTUP18FS','MINIAODMCUP18FS']]
workflows[2018.12] = ['QCD_Pt_80_120_13_UP18', ['QCD_Pt_80_120FS_13_UP18','HARVESTUP18FS','MINIAODMCUP18FS']]
workflows[2018.13] = ['TTbar_13_UP18', ['TTbarFS_13_trackingOnlyValidation_UP18','HARVESTUP18FS_trackingOnly']]

### standard set ###
## particle guns
workflows[15] = ['', ['SingleElectronPt10','DIGI','RECO','HARVEST']]
workflows[16] = ['', ['SingleElectronPt1000','DIGI','RECO','HARVEST']]
workflows[17] = ['', ['SingleElectronPt35','DIGI','RECO','HARVEST']]
workflows[18] = ['', ['SingleGammaPt10','DIGI','RECO','HARVEST']]
workflows[19] = ['', ['SingleGammaPt35','DIGI','RECO','HARVEST']]
workflows[6]  = ['', ['SingleMuPt1','DIGI','RECO','HARVEST']]
workflows[20] = ['', ['SingleMuPt10','DIGI','RECO','HARVEST']]
workflows[21] = ['', ['SingleMuPt100','DIGI','RECO','HARVEST']]
workflows[22] = ['', ['SingleMuPt1000','DIGI','RECO','HARVEST']]
## particle guns postLS1
workflows[1315] = ['', ['SingleElectronPt10_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1316] = ['', ['SingleElectronPt1000_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1317] = ['', ['SingleElectronPt35_UP15','DIGIUP15','RECOUP15','HARVESTUP15','NANOUP15']]
workflows[1318] = ['', ['SingleGammaPt10_UP15','DIGIUP15','RECOUP15','HARVESTUP15','NANOUP15']]
workflows[1319] = ['', ['SingleGammaPt35_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1306]  = ['', ['SingleMuPt1_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1320] = ['', ['SingleMuPt10_UP15','DIGIUP15','RECOUP15','HARVESTUP15','NANOUP15']]
workflows[1321] = ['', ['SingleMuPt100_UP15','DIGIUP15','RECOUP15','HARVESTUP15','NANOUP15']]
workflows[1322] = ['', ['SingleMuPt1000_UP15','DIGIUP15','RECOUP15','HARVESTUP15','NANOUP15']]
workflows[1323] = ['', ['NuGun_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]

## 8 TeV
workflows[24] = ['', ['TTbarLepton','DIGI','RECO','HARVEST']]
workflows[35] = ['', ['Wjet_Pt_80_120','DIGI','RECO','HARVEST']]
workflows[36] = ['', ['Wjet_Pt_3000_3500','DIGI','RECO','HARVEST']]
workflows[37] = ['', ['LM1_sfts','DIGI','RECO','HARVEST']]
# the input for the following worrkflow is high statistics
workflows[38] = ['', ['QCD_FlatPt_15_3000HS','DIGI','RECO','HARVEST']]

workflows[9]  = ['', ['Higgs200ChargedTaus','DIGI','RECO','HARVEST']]
workflows[13] = ['', ['QCD_Pt_3000_3500','DIGI','RECO','HARVEST']]
workflows.addOverride(13,overridesEv5)
workflows[39] = ['', ['QCD_Pt_600_800','DIGI','RECO','HARVEST']]
workflows[23] = ['', ['JpsiMM','DIGI','RECO','HARVEST']]
workflows[25] = ['', ['TTbar','DIGI','RECOAlCaCalo','HARVEST','ALCATT']]
workflows[26] = ['', ['WE','DIGI','RECOAlCaCalo','HARVEST']]
workflows[29] = ['', ['ZEE','DIGI','RECOAlCaCalo','HARVEST']]
workflows[31] = ['', ['ZTT','DIGI','RECO','HARVEST']]
workflows[32] = ['', ['H130GGgluonfusion','DIGI','RECO','HARVEST']]
workflows[33] = ['', ['PhotonJets_Pt_10','DIGI','RECO','HARVEST']]
workflows[34] = ['', ['QQH1352T','DIGI','RECO','HARVEST']]
#workflows[46] = ['', ['ZmumuJets_Pt_20_300']]

workflows[7]  = ['', ['Cosmics','DIGICOS','RECOCOS','ALCACOS','HARVESTCOS']]
workflows[7.1]= ['', ['CosmicsSPLoose','DIGICOS','RECOCOS','ALCACOS','HARVESTCOS']]
workflows[7.20] = ['', ['Cosmics_UP18','DIGICOS_UP18','RECOCOS_UP18','ALCACOS_UP18','HARVESTCOS_UP18']]#2018
workflows[7.21] = ['', ['Cosmics_UP17','DIGICOS_UP17','RECOCOS_UP17','ALCACOS_UP17','HARVESTCOS_UP17']]#2017
workflows[7.22] = ['', ['Cosmics_UP16','DIGICOS_UP16','RECOCOS_UP16','ALCACOS_UP16','HARVESTCOS_UP16']]#2016
workflows[7.23] = ['', ['Cosmics_UP21','DIGICOS_UP21','RECOCOS_UP21','ALCACOS_UP21','HARVESTCOS_UP21']]#2021
workflows[7.24] = ['', ['Cosmics_UP21_0T','DIGICOS_UP21_0T','RECOCOS_UP21_0T','ALCACOS_UP21_0T','HARVESTCOS_UP21_0T']]#2021 0T
workflows[7.3] = ['', ['CosmicsSPLoose_UP18','DIGICOS_UP18','RECOCOS_UP18','ALCACOS_UP18','HARVESTCOS_UP18']]
workflows[7.4] = ['', ['Cosmics_UP18','DIGICOSPEAK_UP18','RECOCOSPEAK_UP18','ALCACOS_UP18','HARVESTCOS_UP18']]

workflows[8]  = ['', ['BeamHalo','DIGICOS','RECOCOS','ALCABH','HARVESTCOS']]
workflows[8.1] = ['', ['BeamHalo_UP18','DIGICOS_UP18','RECOCOS_UP18','ALCABH_UP18','HARVESTCOS_UP18']]
workflows[8.2] = ['', ['BeamHalo_UP21','DIGICOS_UP21','RECOCOS_UP21','ALCABH_UP21','HARVESTCOS_UP21']]

workflows[11] = ['', ['MinBias','DIGI','RECOMIN','HARVEST','ALCAMIN']]
workflows[28] = ['', ['QCD_Pt_80_120','DIGI','RECO','HARVEST']]
workflows[27] = ['', ['WM','DIGI','RECO','HARVEST']]
workflows[30] = ['', ['ZMM','DIGI','RECO','HARVEST']]

workflows[10] = ['', ['ADDMonoJet_d3MD3','DIGI','RECO','HARVEST']]
workflows[12] = ['', ['ZpMM','DIGI','RECO','HARVEST']]
workflows[14] = ['', ['WpM','DIGI','RECO','HARVEST']]

workflows[43] = ['', ['ZpMM_2250_8TeV','DIGI','RECO','HARVEST']]
workflows[44] = ['', ['ZpEE_2250_8TeV','DIGI','RECO','HARVEST']]
workflows[45] = ['', ['ZpTT_1500_8TeV','DIGI','RECO','HARVEST']]

## 13 TeV and postLS1 geometry
workflows[1324] = ['', ['TTbarLepton_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1335] = ['', ['Wjet_Pt_80_120_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1336] = ['', ['Wjet_Pt_3000_3500_13','DIGIUP15','RECOUP15','HARVESTUP15']]
#workflows[1337] = ['', ['LM1_sfts_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1337] = ['', ['SMS-T1tttt_mGl-1500_mLSP-100_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1338] = ['', ['QCD_FlatPt_15_3000HS_13','DIGIUP15','RECOUP15','HARVESTUP15','NANOUP15']]

workflows[1309]  = ['', ['Higgs200ChargedTaus_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1313] = ['', ['QCD_Pt_3000_3500_13','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows.addOverride(1313,overridesEv5)
workflows[1339] = ['', ['QCD_Pt_600_800_13','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[1347] = ['', ['Upsilon1SToMuMu_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1349] = ['', ['BsToMuMu_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1350] = ['', ['JpsiMuMu_Pt-8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1364] = ['', ['BdToMuMu_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1365] = ['', ['BuToJpsiK_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1366] = ['', ['BsToJpsiPhi_13','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[1325] = ['', ['TTbar_13','DIGIUP15','RECOUP15','HARVESTUP15','ALCATTUP15','NANOUP15']]
# the 3 workflows below are for tracking-specific ib test, not to be run in standard relval set.
workflows[1325.1] = ['', ['TTbar_13','DIGIUP15','RECOUP15_trackingOnly','HARVESTUP15_trackingOnly']]
workflows[1325.2] = ['', ['TTbar_13','DIGIUP15','RECOUP15_trackingLowPU','HARVESTUP15']]
workflows[1325.3] = ['', ['TTbar_13','DIGIUP15','RECOUP15_trackingOnlyLowPU','HARVESTUP15_trackingOnly']]
workflows[1325.4] = ['', ['TTbar_13','DIGIUP15','RECOUP15_HIPM','HARVESTUP15']]
# reminiaod wf on 80X MC
workflows[1325.5] = ['', ['ProdZEE_13_reminiaodINPUT','REMINIAOD_mc2016','HARVESTDR2_REMINIAOD_mc2016']]
# reminiaod wf on 94X MC
workflows[1325.51] = ['', ['TTbar_13_94XreminiaodINPUT','REMINIAOD_mc2017','HARVESTUP17_REMINIAOD_mc2017']]
# reminiaod on UL MC
workflows[1325.516] = ['', ['TTbar_13_reminiaod2016UL_preVFP_INPUT','REMINIAOD_mc2016UL_preVFP','HARVESTDR2_REMINIAOD_mc2016UL_preVFP']]
workflows[1325.5161] = ['', ['TTbar_13_reminiaod2016UL_postVFP_INPUT','REMINIAOD_mc2016UL_postVFP','HARVESTDR2_REMINIAOD_mc2016UL_postVFP']]
workflows[1325.517] = ['', ['TTbar_13_reminiaod2017UL_INPUT','REMINIAOD_mc2017UL','HARVESTUP17_REMINIAOD_mc2017UL']]
workflows[1325.518] = ['', ['TTbar_13_reminiaod2018UL_INPUT','REMINIAOD_mc2018UL','HARVESTUP18_REMINIAOD_mc2018UL']]


# nanoaod wf without intermediate EDM,  starting from existing MINIAOD inputs
workflows[1325.6] = ['', ['TTbar_13_94Xv1NanoAODINPUT','NANOAODMC2017_94XMiniAODv1']]
# nanoaod wf with intermediate EDM and merge step, starting from existing MINIAOD inputs
workflows[1325.7] = ['', ['TTbar_13_94Xv2NanoAODINPUT','NANOEDMMC2017_94XMiniAODv2','HARVESTNANOAODMC2017_94XMiniAODv2']]
workflows[1325.8] = ['', ['TTbar_13_94Xv1NanoAODINPUT','NANOEDMMC2017_94XMiniAODv1','HARVESTNANOAODMC2017_94XMiniAODv1']]

# special workflow including the SiStrip APV dynamic gain simulation for 2016 MC
workflows[1325.9] = ['', ['TTbar_13','DIGIUP15APVSimu','RECOUP15','HARVESTUP15','ALCATTUP15']]
workflows[1325.91] = ['', ['TTbar_13','DIGIUP15APVSimu','RECOUP15_HIPM','HARVESTUP15','ALCATTUP15']]

# nanoaod wf without intermediate EDM,  starting from existing MINIAOD inputs
workflows[1325.61] = ['', ['TTbar_13_106Xv1NanoAODINPUT','NANOAODMC2017_106XMiniAODv1']]
# nanoaod wf with intermediate EDM and merge step, starting from existing MINIAOD inputs
workflows[1325.81] = ['', ['TTbar_13_106Xv1NanoAODINPUT','NANOEDMMC2017_106XMiniAODv1','HARVESTNANOAODMC2017_106XMiniAODv1']]

#using ZEE as I cannot find TT at CERN
workflows[1329.1] = ['', ['ZEE_13_80XNanoAODINPUT','NANOEDMMC2016_80X','HARVESTNANOAODMC2016_80X']]

workflows[1326] = ['', ['WE_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1329] = ['', ['ZEE_13','DIGIUP15','RECOUP15_L1TEgDQM','HARVESTUP15_L1TEgDQM','NANOUP15']]

workflows[1356] = ['', ['ZEE_13_DBLMINIAOD','DIGIUP15','RECOAODUP15','HARVESTUP15','DBLMINIAODMCUP15NODQM']] 
workflows[1331] = ['', ['ZTT_13','DIGIUP15','RECOUP15','HARVESTUP15','NANOUP15']]
workflows[1332] = ['', ['H125GGgluonfusion_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1333] = ['', ['PhotonJets_Pt_10_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1334] = ['', ['QQH1352T_13','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[1307] = ['', ['CosmicsSPLoose_UP15','DIGICOS_UP15','RECOCOS_UP15','ALCACOS_UP15','HARVESTCOS_UP15']]
workflows[1308] = ['', ['BeamHalo_13','DIGIHAL','RECOHAL','ALCAHAL','HARVESTHAL']]
workflows[1311] = ['', ['MinBias_13','DIGIUP15','RECOMINUP15','HARVESTMINUP15','ALCAMINUP15']]
workflows[1328] = ['', ['QCD_Pt_80_120_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1327] = ['', ['WM_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1330] = ['', ['ZMM_13','DIGIUP15','RECOUP15_L1TMuDQM','HARVESTUP15_L1TMuDQM','NANOUP15']]

workflows[1310] = ['', ['ADDMonoJet_d3MD3_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1312] = ['', ['ZpMM_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1314] = ['', ['WpM_13','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[1340] = ['', ['PhiToMuMu_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1341] = ['', ['RSKKGluon_m3000GeV_13','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[1343] = ['', ['ZpMM_2250_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1344] = ['', ['ZpEE_2250_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1345] = ['', ['ZpTT_1500_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1348] = ['', ['EtaBToJpsiJpsi_13','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[1351] = ['', ['BuMixing_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1352] = ['', ['HSCPstop_M_200_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1353] = ['', ['RSGravitonToGaGa_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1354] = ['', ['WpToENu_M-2000_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1355] = ['', ['DisplacedSUSY_stopToBottom_M_300_1000mm_13','DIGIUP15','RECOUP15','HARVESTUP15']]

# fullSim 13TeV normal workflows starting from gridpacks LHE generation
workflows[1360] = ['', ['TTbar012Jets_NLO_Mad_py8_Evt_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1361] = ['', ['GluGluHToZZTo4L_M125_Pow_py8_Evt_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1362] = ['', ['VBFHToZZTo4Nu_M125_Pow_py8_Evt_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1363] = ['', ['VBFHToBB_M125_Pow_py8_Evt_13','DIGIUP15','RECOUP15','HARVESTUP15']]

# 2017 workflows starting from gridpacks LHE generation
workflows[1361.17] = ['', ['GluGluHToZZTo4L_M125_Pow_py8_Evt_13UP17','DIGIUP17','RECOUP17','HARVESTUP17','NANOUP17']]
workflows[1362.17] = ['', ['VBFHToZZTo4Nu_M125_Pow_py8_Evt_13UP17','DIGIUP17','RECOUP17','HARVESTUP17','NANOUP17']]
workflows[1363.17] = ['', ['VBFHToBB_M125_Pow_py8_Evt_13UP17','DIGIUP17','RECOUP17','HARVESTUP17','NANOUP17']]

# 2018 workflows starting from gridpacks LHE generation
workflows[1361.18] = ['', ['GluGluHToZZTo4L_M125_Pow_py8_Evt_13UP18','DIGIUP18','RECOUP18','HARVESTUP18','NANOUP18']]
workflows[1362.18] = ['', ['VBFHToZZTo4Nu_M125_Pow_py8_Evt_13UP18','DIGIUP18','RECOUP18','HARVESTUP18','NANOUP18']]
workflows[1363.18] = ['', ['VBFHToBB_M125_Pow_py8_Evt_13UP18','DIGIUP18','RECOUP18','HARVESTUP18','NANOUP18']]

#2018 workflows starting from gridpacks LHE generation with multiple concurrent lumi sections
workflows[1361.181] = ['', ['GluGluHToZZTo4L_M125_Pow_py8_Evt_13UP18ml','DIGIUP18','RECOUP18','HARVESTUP18','NANOUP18']]

# fullSim 13TeV normal workflows starting from pLHE
workflows[1370] = ['', ['GluGluHToGG_M125_Pow_MINLO_NNLOPS_py8_13','Hadronizer_TuneCUETP8M1_13TeV_powhegEmissionVeto_2p_HToGG_M125_13','DIGIUP15','RECOUP15','HARVESTUP15','NANOUP15Had']]

# 2017 workflows starting from pLHE
workflows[1370.17] = ['', ['GluGluHToGG_M125_Pow_MINLO_NNLOPS_py8_13UP17','Hadronizer_TuneCUETP8M1_13TeV_powhegEmissionVeto_2p_HToGG_M125_13UP17','DIGIUP17','RECOUP17','HARVESTUP17','NANOUP17Had']]

# 2018 workflows starting from pLHE
workflows[1370.18] = ['', ['GluGluHToGG_M125_Pow_MINLO_NNLOPS_py8_13UP18','Hadronizer_TuneCUETP8M1_13TeV_powhegEmissionVeto_2p_HToGG_M125_13UP18','DIGIUP18','RECOUP18','HARVESTUP18','NANOUP18Had']]

### HI test ###

### Run I cond., 2011
workflows[140] = ['',['HydjetQ_B12_5020GeV_2011','DIGIHI2011','RECOHI2011','HARVESTHI2011']]
### Run II cond., 2015
workflows[145] = ['',['HydjetQ_B12_5020GeV_2015','DIGIHI2015','RECOHI2015','HARVESTHI2015']]
### Run II cond., 2017
workflows[148] = ['',['HydjetQ_MinBias_XeXe_5442GeV_2017','DIGIHI2017','RECOHI2017','HARVESTHI2017']]
### Run II cond., 2018
workflows[150] = ['',['HydjetQ_B12_5020GeV_2018','DIGIHI2018','RECOHI2018','HARVESTHI2018']]
workflows[158] = ['',['HydjetQ_B12_5020GeV_2018_ppReco','DIGIHI2018PPRECO','RECOHI2018PPRECOMB','ALCARECOHI2018PPRECO','HARVESTHI2018PPRECO']]
workflows[158.01] = ['',['HydjetQ_reminiaodPbPb2018_INPUT','REMINIAODHI2018PPRECOMB','HARVESTHI2018PPRECOMINIAOD']]
workflows[158.1] = ['',['QCD_Pt_80_120_13_HI','DIGIHI2018PPRECO','RECOHI2018PPRECO','HARVESTHI2018PPRECO']]
workflows[158.2] = ['',['PhotonJets_Pt_10_13_HI','DIGIHI2018PPRECO','RECOHI2018PPRECO','HARVESTHI2018PPRECO']]
workflows[158.3] = ['',['ZEEMM_13_HI','DIGIHI2018PPRECO','RECOHI2018PPRECO','HARVESTHI2018PPRECO']]
workflows[159] = ['',['HydjetQ_B12_5020GeV_2021_ppReco','DIGIHI2021PPRECO','RECOHI2021PPRECOMB','ALCARECOHI2021PPRECO','HARVESTHI2021PPRECO']]
workflows[159.1] = ['',['QCD_Pt_80_120_14_HI_2021','DIGIHI2021PPRECO','RECOHI2021PPRECO','HARVESTHI2021PPRECO']]
workflows[159.2] = ['',['PhotonJets_Pt_10_14_HI_2021','DIGIHI2021PPRECO','RECOHI2021PPRECO','HARVESTHI2021PPRECO']]
workflows[159.3] = ['',['ZMM_14_HI_2021','DIGIHI2021PPRECO','RECOHI2021PPRECO','HARVESTHI2021PPRECO']]
workflows[159.4] = ['',['ZEE_14_HI_2021','DIGIHI2021PPRECO','RECOHI2021PPRECO','HARVESTHI2021PPRECO']]
workflows[160] = ['',['Hydjet2Q_MinBias_5020GeV_2018_ppReco','DIGIHI2018PPRECO','RECOHI2018PPRECOMB','ALCARECOHI2018PPRECO','HARVESTHI2018PPRECO']]

### pp reference test ###
workflows[149] = ['',['QCD_Pt_80_120_13_PPREF','DIGIPPREF2017','RECOPPREF2017','HARVESTPPREF2017']]

### pPb test ###
workflows[280]= ['',['AMPT_PPb_5020GeV_MinimumBias','DIGI','RECO','HARVEST']]

### pPb Run2 ###
workflows[281]= ['',['EPOS_PPb_8160GeV_MinimumBias','DIGIUP15_PPb','RECOUP15_PPb','HARVESTUP15_PPb']]
