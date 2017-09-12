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
#workflows[1301] = ['', ['ProdMinBias_13','DIGIUP15PROD1','RECOPRODUP15','MINIAODMCUP15']] # Re-add miniaod here, to keep event content scrutinized
workflows[1301] = ['', ['ProdMinBias_13','DIGIUP15PROD1','RECOPRODUP15']]                  ######################################################## 
workflows[1302] = ['', ['ProdTTbar_13','DIGIUP15PROD1','RECOPRODUP15']]
workflows[1303] = ['', ['ProdQCD_Pt_3000_3500_13','DIGIUP15PROD1','RECOPRODUP15']]
workflows.addOverride(1303,overridesEv5)

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
workflows[134.710] = ['',['RunZeroBias2015B','HLTDR2_50ns','RECODR2_50nsreHLT_HIPM','HARVESTDR2']]

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
workflows[134.812] = ['',['RunZeroBias2015C','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]
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
workflows[134.912] = ['',['RunZeroBias2015D','HLTDR2_25ns','RECODR2_25nsreHLT_HIPM','HARVESTDR2']]

### run 2016B ###
workflows[136.721] = ['',['RunHLTPhy2016B','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.722] = ['',['RunDoubleEG2016B','HLTDR2_2016','RECODR2_2016reHLT_skimDoubleEG_HIPM','HARVESTDR2']]
workflows[136.723] = ['',['RunDoubleMuon2016B','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.724] = ['',['RunJetHT2016B','HLTDR2_2016','RECODR2_2016reHLT_skimJetHT_HIPM','HARVESTDR2']]
workflows[136.725] = ['',['RunMET2016B','HLTDR2_2016','RECODR2_2016reHLT_skimMET_HIPM','HARVESTDR2']]
workflows[136.726] = ['',['RunMuonEG2016B','HLTDR2_2016','RECODR2_2016reHLT_skimMuonEG_HIPM','HARVESTDR2']]
workflows[136.727] = ['',['RunDoubleEGPrpt2016B','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.728] = ['',['RunSingleMuPrpt2016B','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.729] = ['',['RunSingleEl2016B','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.730] = ['',['RunSingleMu2016B','HLTDR2_2016','RECODR2_2016reHLT_skimSingleMu_HIPM','HARVESTDR2']]
workflows[136.731] = ['',['RunSinglePh2016B','HLTDR2_2016','RECODR2_2016reHLT_skimSinglePh_HIPM','HARVESTDR2']]
workflows[136.732] = ['',['RunZeroBias2016B','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.733] = ['',['RunCosmics2016B','RECOCOSDRUN2','ALCACOSDRUN2','HARVESTDCRUN2']]
workflows[136.734] = ['',['RunMuOnia2016B','HLTDR2_2016','RECODR2_2016reHLT_skimMuOnia_HIPM','HARVESTDR2']]
workflows[136.735] = ['',['RunNoBPTX2016B','HLTDR2_2016','RECODR2reHLTAlCaTkCosmics_HIPM','HARVESTDR2']]
workflows[136.7321] = ['',['RunZeroBias2016BnewL1repack','HLTDR2newL1repack_2016','RECODR2newL1repack_2016reHLT_HIPM','HARVESTDR2']]

### run 2016C ###
workflows[136.736] = ['',['RunHLTPhy2016C','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.737] = ['',['RunDoubleEG2016C','HLTDR2_2016','RECODR2_2016reHLT_skimDoubleEG_HIPM','HARVESTDR2']]
workflows[136.738] = ['',['RunDoubleMuon2016C','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.739] = ['',['RunJetHT2016C','HLTDR2_2016','RECODR2_2016reHLT_skimJetHT_HIPM','HARVESTDR2']]
workflows[136.740] = ['',['RunMET2016C','HLTDR2_2016','RECODR2_2016reHLT_skimMET_HIPM','HARVESTDR2']]
workflows[136.741] = ['',['RunMuonEG2016C','HLTDR2_2016','RECODR2_2016reHLT_skimMuonEG_HIPM','HARVESTDR2']]
workflows[136.742] = ['',['RunSingleEl2016C','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.743] = ['',['RunSingleMu2016C','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.744] = ['',['RunSinglePh2016C','HLTDR2_2016','RECODR2_2016reHLT_skimSinglePh_HIPM','HARVESTDR2']]
workflows[136.745] = ['',['RunZeroBias2016C','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.746] = ['',['RunMuOnia2016C','HLTDR2_2016','RECODR2_2016reHLT_skimMuOnia_HIPM','HARVESTDR2']]

### run 2016D ###
workflows[136.747] = ['',['RunHLTPhy2016D','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.748] = ['',['RunDoubleEG2016D','HLTDR2_2016','RECODR2_2016reHLT_skimDoubleEG_HIPM','HARVESTDR2']]
workflows[136.749] = ['',['RunDoubleMuon2016D','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.750] = ['',['RunJetHT2016D','HLTDR2_2016','RECODR2_2016reHLT_skimJetHT_HIPM','HARVESTDR2']]
workflows[136.751] = ['',['RunMET2016D','HLTDR2_2016','RECODR2_2016reHLT_skimMET_HIPM','HARVESTDR2']]
workflows[136.752] = ['',['RunMuonEG2016D','HLTDR2_2016','RECODR2_2016reHLT_skimMuonEG_HIPM','HARVESTDR2']]
workflows[136.753] = ['',['RunSingleEl2016D','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.754] = ['',['RunSingleMu2016D','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.755] = ['',['RunSinglePh2016D','HLTDR2_2016','RECODR2_2016reHLT_skimSinglePh_HIPM','HARVESTDR2']]
workflows[136.756] = ['',['RunZeroBias2016D','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.757] = ['',['RunMuOnia2016D','HLTDR2_2016','RECODR2_2016reHLT_skimMuOnia_HIPM','HARVESTDR2']]

### run 2016E ###
workflows[136.758] = ['',['RunHLTPhy2016E','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.759] = ['',['RunDoubleEG2016E','HLTDR2_2016','RECODR2_2016reHLT_skimDoubleEG_HIPM','HARVESTDR2']]
workflows[136.760] = ['',['RunDoubleMuon2016E','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.761] = ['',['RunJetHT2016E','HLTDR2_2016','RECODR2_2016reHLT_skimJetHT_HIPM','HARVESTDR2']]
workflows[136.762] = ['',['RunMET2016E','HLTDR2_2016','RECODR2_2016reHLT_skimMET_HIPM','HARVESTDR2']]
workflows[136.763] = ['',['RunMuonEG2016E','HLTDR2_2016','RECODR2_2016reHLT_skimMuonEG_HIPM','HARVESTDR2']]
workflows[136.764] = ['',['RunSingleEl2016E','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.765] = ['',['RunSingleMu2016E','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.766] = ['',['RunSinglePh2016E','HLTDR2_2016','RECODR2_2016reHLT_skimSinglePh_HIPM','HARVESTDR2']]
workflows[136.767] = ['',['RunZeroBias2016E','HLTDR2_2016','RECODR2_2016reHLT_HIPM','HARVESTDR2']]
workflows[136.768] = ['',['RunMuOnia2016E','HLTDR2_2016','RECODR2_2016reHLT_skimMuOnia_HIPM','HARVESTDR2']]
# reminiAOD wf on 2016E 80X input
workflows[136.7611] = ['',['RunJetHT2016E_reminiaod','REMINIAOD_data2016','HARVESTDR2_REMINIAOD_data2016']]

### run 2016H ###
workflows[136.769] = ['',['RunHLTPhy2016H','HLTDR2_2016','RECODR2_2016reHLT_Prompt','HARVESTDR2']]
workflows[136.770] = ['',['RunDoubleEG2016H','HLTDR2_2016','RECODR2_2016reHLT_skimDoubleEG_Prompt','HARVESTDR2']]
workflows[136.771] = ['',['RunDoubleMuon2016H','HLTDR2_2016','RECODR2_2016reHLT_Prompt','HARVESTDR2']]
workflows[136.772] = ['',['RunJetHT2016H','HLTDR2_2016','RECODR2_2016reHLT_skimJetHT_Prompt','HARVESTDR2']]
workflows[136.773] = ['',['RunMET2016H','HLTDR2_2016','RECODR2_2016reHLT_skimMET_Prompt','HARVESTDR2']]
workflows[136.774] = ['',['RunMuonEG2016H','HLTDR2_2016','RECODR2_2016reHLT_skimMuonEG_Prompt','HARVESTDR2']]
workflows[136.775] = ['',['RunSingleEl2016H','HLTDR2_2016','RECODR2_2016reHLT_Prompt','HARVESTDR2']]
workflows[136.776] = ['',['RunSingleMu2016H','HLTDR2_2016','RECODR2_2016reHLT_Prompt_Lumi','HARVESTDR2']]
workflows[136.777] = ['',['RunSinglePh2016H','HLTDR2_2016','RECODR2_2016reHLT_skimSinglePh_Prompt','HARVESTDR2']]
workflows[136.778] = ['',['RunZeroBias2016H','HLTDR2_2016','RECODR2_2016reHLT_Prompt','HARVESTDR2']]
workflows[136.779] = ['',['RunMuOnia2016H','HLTDR2_2016','RECODR2_2016reHLT_skimMuOnia_Prompt','HARVESTDR2']]

### run 2017B ###
workflows[136.780] = ['',['RunHLTPhy2017B','HLTDR2_2017','RECODR2_2017reHLT_Prompt','HARVEST2017']]
workflows[136.781] = ['',['RunDoubleEG2017B','HLTDR2_2017','RECODR2_2017reHLT_skimDoubleEG_Prompt','HARVEST2017']]
workflows[136.782] = ['',['RunDoubleMuon2017B','HLTDR2_2017','RECODR2_2017reHLT_Prompt','HARVEST2017']]
workflows[136.783] = ['',['RunJetHT2017B','HLTDR2_2017','RECODR2_2017reHLT_skimJetHT_Prompt','HARVEST2017']]
workflows[136.784] = ['',['RunMET2017B','HLTDR2_2017','RECODR2_2017reHLT_skimMET_Prompt','HARVEST2017']]
workflows[136.785] = ['',['RunMuonEG2017B','HLTDR2_2017','RECODR2_2017reHLT_skimMuonEG_Prompt','HARVEST2017']]
workflows[136.786] = ['',['RunSingleEl2017B','HLTDR2_2017','RECODR2_2017reHLT_Prompt','HARVEST2017']]
workflows[136.787] = ['',['RunSingleMu2017B','HLTDR2_2017','RECODR2_2017reHLT_skimSingleMu_Prompt_Lumi','HARVEST2017']]
workflows[136.788] = ['',['RunSinglePh2017B','HLTDR2_2017','RECODR2_2017reHLT_skimSinglePh_Prompt','HARVEST2017']]
workflows[136.789] = ['',['RunZeroBias2017B','HLTDR2_2017','RECODR2_2017reHLT_Prompt','HARVEST2017']]
workflows[136.790] = ['',['RunMuOnia2017B','HLTDR2_2017','RECODR2_2017reHLT_skimMuOnia_Prompt','HARVEST2017']]
workflows[136.791] = ['',['RunNoBPTX2017B','HLTDR2_2017','RECODR2_2017reHLTAlCaTkCosmics_Prompt','HARVEST2017']]


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
workflows[1317] = ['', ['SingleElectronPt35_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1318] = ['', ['SingleGammaPt10_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1319] = ['', ['SingleGammaPt35_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1306]  = ['', ['SingleMuPt1_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1320] = ['', ['SingleMuPt10_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1321] = ['', ['SingleMuPt100_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1322] = ['', ['SingleMuPt1000_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
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
workflows[7.2] = ['', ['Cosmics_UP17','DIGICOS_UP17','RECOCOS_UP17','ALCACOS_UP17','HARVESTCOS_UP17']]
workflows[7.3] = ['', ['CosmicsSPLoose_UP17','DIGICOS_UP17','RECOCOS_UP17','ALCACOS_UP17','HARVESTCOS_UP17']]
workflows[7.4] = ['', ['Cosmics_UP17','DIGICOSPEAK_UP17','RECOCOSPEAK_UP17','ALCACOS_UP17','HARVESTCOS_UP17']]

workflows[8]  = ['', ['BeamHalo','DIGICOS','RECOCOS','ALCABH','HARVESTCOS']]
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
workflows[1338] = ['', ['QCD_FlatPt_15_3000HS_13','DIGIUP15','RECOUP15','HARVESTUP15']]

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

workflows[1325] = ['', ['TTbar_13','DIGIUP15','RECOUP15','HARVESTUP15','ALCATTUP15']]
# the 3 workflows below are for tracking-specific ib test, not to be run in standard relval set.
workflows[1325.1] = ['', ['TTbar_13','DIGIUP15','RECOUP15_trackingOnly','HARVESTUP15_trackingOnly']]
workflows[1325.2] = ['', ['TTbar_13','DIGIUP15','RECOUP15_trackingLowPU','HARVESTUP15']]
workflows[1325.3] = ['', ['TTbar_13','DIGIUP15','RECOUP15_trackingOnlyLowPU','HARVESTUP15_trackingOnly']]
workflows[1325.4] = ['', ['TTbar_13','DIGIUP15','RECOUP15_HIPM','HARVESTUP15']]
# reminiaod wf on 80X MC
workflows[1325.5] = ['', ['TTbar_13_reminiaodINPUT','REMINIAOD_mc2016','HARVESTDR2_REMINIAOD_mc2016']]

workflows[1326] = ['', ['WE_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1329] = ['', ['ZEE_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1356] = ['', ['ZEE_13_DBLMINIAOD','DIGIUP15','RECOAODUP15','HARVESTUP15','DBLMINIAODMCUP15NODQM']] 
workflows[1331] = ['', ['ZTT_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1332] = ['', ['H125GGgluonfusion_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1333] = ['', ['PhotonJets_Pt_10_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1334] = ['', ['QQH1352T_13','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[1307] = ['', ['CosmicsSPLoose_UP15','DIGICOS_UP15','RECOCOS_UP15','ALCACOS_UP15','HARVESTCOS_UP15']]
workflows[1308] = ['', ['BeamHalo_13','DIGIHAL','RECOHAL','ALCAHAL','HARVESTHAL']]
workflows[1311] = ['', ['MinBias_13','DIGIUP15','RECOMINUP15','HARVESTMINUP15','ALCAMINUP15']]
workflows[1328] = ['', ['QCD_Pt_80_120_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1327] = ['', ['WM_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1330] = ['', ['ZMM_13','DIGIUP15','RECOUP15','HARVESTUP15']]

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
workflows[1361.17] = ['', ['GluGluHToZZTo4L_M125_Pow_py8_Evt_13UP17','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[1362.17] = ['', ['VBFHToZZTo4Nu_M125_Pow_py8_Evt_13UP17','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[1363.17] = ['', ['VBFHToBB_M125_Pow_py8_Evt_13UP17','DIGIUP17','RECOUP17','HARVESTUP17']]

### HI test ###

### Run I cond., 2011
workflows[140] = ['',['HydjetQ_B12_5020GeV_2011','DIGIHI2011','RECOHI2011','HARVESTHI2011']]
### Run II cond., 2015
workflows[145] = ['',['HydjetQ_B12_5020GeV_2015','DIGIHI2015','RECOHI2015','HARVESTHI2015']]
### Run II cond., 2018
workflows[150] = ['',['HydjetQ_B12_5020GeV_2018','DIGIHI2018','RECOHI2018','HARVESTHI2018']]
workflows[150.1] = ['',['QCD_Pt_80_120_13_HI','DIGIHI2018','RECOHI2018','HARVESTHI2018']]
workflows[150.2] = ['',['PhotonJets_Pt_10_13_HI','DIGIHI2018','RECOHI2018','HARVESTHI2018']]
workflows[150.3] = ['',['ZEEMM_13_HI','DIGIHI2018','RECOHI2018','HARVESTHI2018']]

### pPb test ###
workflows[280]= ['',['AMPT_PPb_5020GeV_MinimumBias','DIGI','RECO','HARVEST']]

### pPb Run2 ###
workflows[281]= ['',['EPOS_PPb_8160GeV_MinimumBias','DIGIUP15_PPb','RECOUP15_PPb','HARVESTUP15_PPb']]
