#!/bin/bash

# python ~piet/public/das_client.py --query="dataset=/DYToMuMu*/amkalsi*RAW*/* instance=prod/phys03" --limit=100
# python das_client.py --query="dataset=/DYToMuMu*/amkalsi*RAW*/* instance=prod/phys03" --limit=100

# python das_client.py --query="summary dataset=/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO-a6c1ab73bd1959e4a7fbbca874362562/USER instance=prod/phys03" --limit=100

# python das_client.py --query="file dataset=/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO-a6c1ab73bd1959e4a7fbbca874362562/USER instance=prod/phys03" --limit=100

python das_client.py --query="dataset=/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_*Neutr*_RECO-*/USER instance=prod/phys03" --limit=100

#### Datasets Piet ####
#######################
# python das_client.py --query="dataset=/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_*v2*_RECO-*/USER instance=prod/phys03" --limit=100
# echo "------------------------"
# for i in `python das_client.py --query="dataset=/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_*v2*_RECO-*/USER instance=prod/phys03" --limit=100 | grep DYToMuMu`; do
# echo ""
# echo "Summary for Dataset ${i}"
# python das_client.py --query="summary dataset=${i} instance=prod/phys03" --limit=100
# echo ""
# done
# echo "------------------------"
#######################

### Datasets Cesare ###
#######################
# export DataCesare="/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/calabria-calabria_DYToMuMu_GEN-SIM-RECO_CMSSW_6_2_0_SLHC23patch1_2023_3Step_OKFS3-2dad437730bcb898314ced9a1ae33ee0/USER"
# echo "Summary for Dataset ${DataCesare}" 
# python das_client.py --query="summary dataset=${DataCesare} instance=prod/phys03" --limit=100
# echo "First 10 files for Dataset ${DataCesare}"
# echo "------------------------"
# python das_client.py --query="file dataset=${DataCesare} instance=prod/phys03" --limit=10
# echo "------------------------"
# echo "Parent files for first 10 files for Dataset ${DataCesare}"
# echo "------------------------"
# for i in `python das_client.py --query="file dataset=${DataCesare} instance=prod/phys03" --limit=10 | grep root`; do
#     python das_client.py --query="parent file=${i} instance=prod/phys03" --limit=10
# done
# echo "------------------------"

### Datasets David ####
#######################
# export DataDavid="/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/dnash-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_2023SHCalNoTaper_PU140_Selectors_RECO-b52ce42d5986c94dc336f39e015d825e/USER"
# echo "Summary for Dataset ${DataDavid}"
# python das_client.py --query="summary dataset=${DataDavid} instance=prod/phys03" --limit=100
# echo "First 10 files for Dataset ${DataDavid}"
# python das_client.py --query="file dataset=${DataDavid} instance=prod/phys03" --limit=10

# python das_client.py --query="parent file=/store/group/upgrade/muon/ME0GlobalReco/ME0MuonReRun_DY_SLHC23patch1_SegmentReRunFullRun_ForPublish/M-20_TuneZ2star_14TeV_6_2_0_SLHC23patch1_2023/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_2023SHCalNoTaper_PU140_Selectors_RECO/b52ce42d5986c94dc336f39e015d825e/output_896_2_3B7.root instance=prod/phys03" --limit=10

