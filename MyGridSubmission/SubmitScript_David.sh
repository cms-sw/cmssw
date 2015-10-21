#!/bin/bash


### Datasets 2B run upon   ### 
##############################
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_50ps_1ns_RECO-97dad57322a4294bc9a4ba07e1737a01/USER
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO-a6c1ab73bd1959e4a7fbbca874362562/USER
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_RECO-c7e8aa97dc89be76b8287a47f368c033/USER
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_500ps_4ns_RECO-b08719edd4307d7022706f7e10bafa83/USER
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_RECO-dae2ba7355add6539efc58ac0d4cc16b/USER
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_32ns_RECO-9534beda007a59447d9159ddc4f0a675/USER

# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_50ps-330cdf405e793f966b0b9f5860b4f48a/USER
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_100ps-42dac2b4e5826089ee9814ac4d670486/USER
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_300ps-65914b296c27131144943ce7f41d276e/USER
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_500ps-21addc7e1c7d007a3fb32c764bd2f4dd/USER
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_1ns-a78e67d4dfe6ec91a4b2ef7c9deeaea8/USER
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_5ns-d3908b1662dce7c3d94d8e9843b3ae74/USER


### Submission with CRAB 3 ### 
##############################
### https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookCRAB3Tutorial
### (not possible with SLHC SCRAM_ARCH = 472)
##############################

crab submit -c myCrab3crabConfig_David.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_50ps_1ns_RECO-97dad57322a4294bc9a4ba07e1737a01/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_50ps_1ns_Tight_Analysis_David' JobType.psetName='testME0MuonAnalyzer_Tight_cfg.py'

crab submit -c myCrab3crabConfig_David.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO-a6c1ab73bd1959e4a7fbbca874362562/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_Tight_Analysis_David' JobType.psetName='testME0MuonAnalyzer_Tight_cfg.py'

crab submit -c myCrab3crabConfig_David.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_RECO-c7e8aa97dc89be76b8287a47f368c033/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_Tight_Analysis_David' JobType.psetName='testME0MuonAnalyzer_Tight_cfg.py'

crab submit -c myCrab3crabConfig_David.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_500ps_4ns_RECO-b08719edd4307d7022706f7e10bafa83/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_500ps_4ns_Tight_Analysis_David' JobType.psetName='testME0MuonAnalyzer_Tight_cfg.py'

crab submit -c myCrab3crabConfig_David.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_RECO-dae2ba7355add6539efc58ac0d4cc16b/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_Tight_Analysis_David' JobType.psetName='testME0MuonAnalyzer_Tight_cfg.py'

crab submit -c myCrab3crabConfig_David.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_32ns_RECO-9534beda007a59447d9159ddc4f0a675/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_32ns_Tight_Analysis_David' JobType.psetName='testME0MuonAnalyzer_Tight_cfg.py'

crab submit -c myCrab3crabConfig_David.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_50ps_1ns_RECO-97dad57322a4294bc9a4ba07e1737a01/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_50ps_1ns_Loose_Analysis_David' JobType.psetName='testME0MuonAnalyzer_Loose_cfg.py'

crab submit -c myCrab3crabConfig_David.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO-a6c1ab73bd1959e4a7fbbca874362562/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_Loose_Analysis_David' JobType.psetName='testME0MuonAnalyzer_Loose_cfg.py'

crab submit -c myCrab3crabConfig_David.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_RECO-c7e8aa97dc89be76b8287a47f368c033/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_Loose_Analysis_David' JobType.psetName='testME0MuonAnalyzer_Loose_cfg.py'

crab submit -c myCrab3crabConfig_David.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_500ps_4ns_RECO-b08719edd4307d7022706f7e10bafa83/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_500ps_4ns_Loose_Analysis_David' JobType.psetName='testME0MuonAnalyzer_Loose_cfg.py'

crab submit -c myCrab3crabConfig_David.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_RECO-dae2ba7355add6539efc58ac0d4cc16b/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_Loose_Analysis_David' JobType.psetName='testME0MuonAnalyzer_Loose_cfg.py'

crab submit -c myCrab3crabConfig_David.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_32ns_RECO-9534beda007a59447d9159ddc4f0a675/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_32ns_Loose_Analysis_David' JobType.psetName='testME0MuonAnalyzer_Loose_cfg.py'
