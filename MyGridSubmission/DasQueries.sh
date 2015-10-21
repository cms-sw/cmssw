#!/bin/bash

# python ~piet/public/das_client.py --query="dataset=/DYToMuMu*/amkalsi*RAW*/* instance=prod/phys03" --limit=100
# python das_client.py --query="dataset=/DYToMuMu*/amkalsi*RAW*/* instance=prod/phys03" --limit=100

# python das_client.py --query="summary dataset=/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO-a6c1ab73bd1959e4a7fbbca874362562/USER instance=prod/phys03" --limit=100

# python das_client.py --query="file dataset=/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO-a6c1ab73bd1959e4a7fbbca874362562/USER instance=prod/phys03" --limit=100

python das_client.py --query="dataset=/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/piet-DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_*_RECO-*/USER instance=prod/phys03" --limit=100

