#!/bin/bash


### Datasets 2B run upon   ### 
##############################
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_50ps-330cdf405e793f966b0b9f5860b4f48a/USER
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_100ps-42dac2b4e5826089ee9814ac4d670486/USER
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_300ps-65914b296c27131144943ce7f41d276e/USER
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_500ps-21addc7e1c7d007a3fb32c764bd2f4dd/USER
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_1ns-a78e67d4dfe6ec91a4b2ef7c9deeaea8/USER
# /DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_5ns-d3908b1662dce7c3d94d8e9843b3ae74/USER


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


### Submission with CRAB 3 ### 
##############################
### https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookCRAB3Tutorial
### (not possible with SLHC SCRAM_ARCH = 472)
##############################

# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_100ps-42dac2b4e5826089ee9814ac4d670486/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO'
# not submitted, submitted with crab2

# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_300ps-65914b296c27131144943ce7f41d276e/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_RECO'
# not submitted, submitted with crab2

crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_50ps-330cdf405e793f966b0b9f5860b4f48a/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_50ps_1ns_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_50ps_1ns_RECO'

# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_500ps-21addc7e1c7d007a3fb32c764bd2f4dd/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_500ps_4ns_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_500ps_4ns_RECO'

# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_1ns-a78e67d4dfe6ec91a4b2ef7c9deeaea8/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_RECO'

# crab submit -c myCrab3crabConfig.py Data.inputDataset='/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/amkalsi-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_MinBias_ME0_RAW_TimingResolution_5ns-d3908b1662dce7c3d94d8e9843b3ae74/USER' General.requestName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_32ns_RECO' Data.publishDataName='DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_32ns_RECO'


