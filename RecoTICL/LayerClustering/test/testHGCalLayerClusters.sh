#!/bin/bash
# The layer clustering of known rechits must pass every check, and must fail when the layer sizes do not add up to
# the number of rechits.
CFG=${SCRAM_TEST_PATH:-${LOCALTOP}/src/RecoTICL/LayerClustering/test}/testHGCalLayerClusters_cfg.py
cmsRun ${CFG} || exit 1
cmsRun ${CFG} badLayerSizes > badLayerSizes.log 2>&1 && { echo "the job with bad layer sizes did not fail"; exit 1; }
grep -q "the layer sizes do not add up" badLayerSizes.log || { cat badLayerSizes.log; exit 1; }
echo "OK"
