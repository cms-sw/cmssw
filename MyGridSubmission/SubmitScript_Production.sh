#!/bin/bash


### Datasets 2B run upon   ### 
##############################
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
# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_50ps-330cdf405e793f966b0b9f5860b4f48a/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_50ps_1ns_v2_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_50ps_1ns_v2_RECO'
# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_100ps-42dac2b4e5826089ee9814ac4d670486/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_v2_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_v2_RECO'
# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_300ps-65914b296c27131144943ce7f41d276e/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_v2_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_v2_RECO'
# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_500ps-21addc7e1c7d007a3fb32c764bd2f4dd/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_500ps_4ns_v2_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_500ps_4ns_v2_RECO'
# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_1ns-a78e67d4dfe6ec91a4b2ef7c9deeaea8/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_v2_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_v2_RECO'                                                                                                                                                                         # crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_5ns-d3908b1662dce7c3d94d8e9843b3ae74/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_32ns_v2_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_32ns_v2_RECO'                      
# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_5ns-d3908b1662dce7c3d94d8e9843b3ae74/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_45ns_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_45ns_RECO'
# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_5ns-d3908b1662dce7c3d94d8e9843b3ae74/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_60ns_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_60ns_RECO'
# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_5ns-d3908b1662dce7c3d94d8e9843b3ae74/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_90ns_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_90ns_RECO'
# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_5ns-d3908b1662dce7c3d94d8e9843b3ae74/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_150ns_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_150ns_RECO'

### Neutron Background Added #####################################################
# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_100ps-42dac2b4e5826089ee9814ac4d670486/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_NeutrBkg_5E34_v2_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_NeutrBkg_5E34_v2_RECO'

# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_1ns-a78e67d4dfe6ec91a4b2ef7c9deeaea8/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_NeutrBkg_5E34_v2_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_NeutrBkg_5E34_v2_RECO'

# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_5ns-d3908b1662dce7c3d94d8e9843b3ae74/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_32ns_NeutrBkg_5E34_v2_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_32ns_NeutrBkg_5E34_v2_RECO'


### Neutron Background recalculated ... new DIGI step  ###########################
### Datasets to be used: 
### signal: /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/TP2023HGCALGS-newsplit_DES23_62_V1-v1/GEN-SIM [ 1M,  253 files, T2_US_Vanderbilt]
### pileup: /MinBias_TuneZ2star_14TeV-pythia6/TP2023HGCALGS-DES23_62_V1-v3/GEN-SIM                       [10M, 1094 files, T2_CH_CERN, T2_FR_GRIF_LLR, T2_US_Vanderbilt]
##################################################################################
### Submit samples with different time resolution                        
#
crab submit -c myCrab3crabConfig_DIGI.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/TP2023HGCALGS-newsplit_DES23_62_V1-v1/GEN-SIM' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_500um_1cm_v2_DIGI' Data.outputDatasetTag='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_500um_1cm_v2_DIGI'
#
# crab submit -c myCrab3crabConfig_DIGI.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/TP2023HGCALGS-newsplit_DES23_62_V1-v1/GEN-SIM' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_500um_1cm_v2_DIGI' Data.outputDatasetTag='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_500um_1cm_v2_DIGI'
#
# crab submit -c myCrab3crabConfig_DIGI.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/TP2023HGCALGS-newsplit_DES23_62_V1-v1/GEN-SIM' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_500ps_500um_1cm_v2_DIGI' Data.outputDatasetTag='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_500ps_500um_1cm_v2_DIGI'
#
# crab submit -c myCrab3crabConfig_DIGI.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/TP2023HGCALGS-newsplit_DES23_62_V1-v1/GEN-SIM' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_500um_1cm_v2_DIGI' Data.outputDatasetTag='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_500um_1cm_v2_DIGI'
#
# crab submit -c myCrab3crabConfig_DIGI.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/TP2023HGCALGS-newsplit_DES23_62_V1-v1/GEN-SIM' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_500um_1cm_v2_DIGI' Data.outputDatasetTag='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_500um_1cm_v2_DIGI'
# 
### Submit samples with different spatial resolution
#
# crab submit -c myCrab3crabConfig_DIGI.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/TP2023HGCALGS-newsplit_DES23_62_V1-v1/GEN-SIM' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_100um_1cm_v2_DIGI' Data.outputDatasetTag='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_100um_1cm_v2_DIGI'
# 
# crab submit -c myCrab3crabConfig_DIGI.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/TP2023HGCALGS-newsplit_DES23_62_V1-v1/GEN-SIM' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_250um_1cm_v2_DIGI' Data.outputDatasetTag='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_250um_1cm_v2_DIGI'
# 
# crab submit -c myCrab3crabConfig_DIGI.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/TP2023HGCALGS-newsplit_DES23_62_V1-v1/GEN-SIM' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1mm_1cm_v2_DIGI' Data.outputDatasetTag='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1mm_1cm_v2_DIGI'
#
###  signal: /ZprimeSSMToMuMu_M2500_TuneZ2star_14TeV_pythia6_cfg_ME0_gensim/rosma-ZprimeSSMToMuMu_M2500_TuneZ2star_14TeV_pythia6_cfg_ME0_gensim-a7e94321ca0f14f2d615fb718b2cab17/USER [100K, 10K files, T2_IT_Bari]
###  pileup: /MinBias_TuneZ2star_14TeV-pythia6/TP2023HGCALGS-DES23_62_V1-v3/GEN-SIM                                                                                                   [10M, 1094 files, T2_CH_CERN, T2_FR_GRIF_LLR, T2_US_Vanderbilt]
# 
# crab submit -c myCrab3crabConfig_DIGI.py Data.inputDataset='/ZprimeSSMToMuMu_M2500_TuneZ2star_14TeV_pythia6_cfg_ME0_gensim/rosma-ZprimeSSMToMuMu_M2500_TuneZ2star_14TeV_pythia6_cfg_ME0_gensim-a7e94321ca0f14f2d615fb718b2cab17/USER' General.requestName='ZprimeSSMToMuMu_M2500_TuneZ2star_14TeV_pythia6_HGCALS_PU140_1ns_100um_1cm_v2_DIGI' Data.outputDatasetTag='ZprimeSSMToMuMu_M2500_TuneZ2star_14TeV_pythia6_HGCALGS_PU140_1ns_100um_1cm_v2_DIGI'





### Submission with CRAB 2 ###
##############################
### https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookCRAB2Tutorial
##############################

# crab -cfg myCrab2crabConfig.cfg -create
# crab -c DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO -submit 1-200 
# crab -c DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO -submit 201-400 
# crab -c DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO -submit 401-600
# crab -c DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO -submit 601-800 
# crab -c DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO -submit 801-1000 
# submitted and published

# crab -cfg myCrab2crabConfig.cfg -create
# crab -c DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_RECO -submit 1-200 
# crab -c DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_RECO -submit 201-400 
# crab -c DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_RECO -submit 401-600
# crab -c DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_RECO -submit 601-800 
# crab -c DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_RECO -submit 801-1000 
# submitted

# crab -cfg myCrab2crabConfig.cfg -create
# crab -c DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_RECO -submit 1-200 
# crab -c DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_RECO -submit 201-400 
# crab -c DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_RECO -submit 401-600
# crab -c DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_RECO -submit 601-800 
# crab -c DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_RECO -submit 801-1000 
# failed to submit


