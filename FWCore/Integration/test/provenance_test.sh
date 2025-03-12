#!/bin/sh


function die { echo $1: status $2 ;  exit $2; }

#The two jobs will have different ProductRegistries in their output files but have the same ProcessHistory.
# The ProductRegistry just differ because the internal dependencies between the data products is different
# and PoolOutputModule only stores provenance of 'dropped' data products IFF they are parents of a kept product. 
# The check makes sure the provenance in the ProductRegistry is properly updated when the new file is read
cmsRun ${SCRAM_TEST_PATH}/provenance_prod_cfg.py || die 'Failed in provenance_prod_cfg.py' $?
cmsRun ${SCRAM_TEST_PATH}/provenance_prod_cfg.py  --consumeProd2 || die 'Failed in provenance_prod_cfg.py  --consumeProd2' $?
cmsRun ${SCRAM_TEST_PATH}/provenance_check_cfg.py || die 'Failed test of provenance' $?