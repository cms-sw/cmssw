#!/bin/bash

# argument is config file
source /nfshome0/ecallaser/config/lmf_cfg
source ~cmssw2/scripts/setup.sh
cd ${MON_CMSSW_REL_DIR}
eval `scramv1 runtime -sh`
source ${MON_MUSECAL_DIR}/setup /nfshome0/ecallaser/config/lmf_cfg
source ${MON_MUSECAL_DIR}/freezeLM.pl $1 $2

