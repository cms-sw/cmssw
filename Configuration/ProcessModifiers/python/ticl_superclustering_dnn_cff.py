import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
from Configuration.ProcessModifiers.ticl_superclustering_mustache_pf_cff import ticl_superclustering_mustache_pf
from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl

# Modifier to run superclustering with DNN
# It is not to be used directly in a Process since it is currently the default for ticl_v5
# It only exists as convenience in configuration files 
ticl_superclustering_dnn = ticl_v5 & (~ticl_superclustering_mustache_pf) & (~ticl_superclustering_mustache_ticl)
