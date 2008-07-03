#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

(cmsRun ${LOCAL_TEST_DIR}/test_es_prefer_prods_cfg.py ) || die 'Failure using test_es_prefer_prods_cfg.py' $?
(cmsRun ${LOCAL_TEST_DIR}/test_es_prefer_sources_cfg.py ) || die 'Failure using test_es_prefer_sources_cfg.py' $?
(cmsRun ${LOCAL_TEST_DIR}/test_es_prefer_source_beats_prod_cfg.py ) || die 'Failure using test_es_prefer_source_beats_prod_cfg.py' $?
(cmsRun ${LOCAL_TEST_DIR}/test_prod_trumps_source_cfg.py ) || die 'Failure using test_prod_trumps_source_cfg.py' $?
(cmsRun ${LOCAL_TEST_DIR}/test_es_prefer_2_es_sources_order1_cfg.py ) || die 'Failure using test_es_prefer_2_es_sources_order1_cfg.py' $?
(cmsRun ${LOCAL_TEST_DIR}/test_es_prefer_2_es_sources_order2_cfg.py ) || die 'Failure using test_es_prefer_2_es_sources_order2_cfg.py' $?
!(cmsRun ${LOCAL_TEST_DIR}/test_2_es_sources_no_prefer_cfg.py ) || die 'Should have failed using test_2_es_sources_no_prefer_cfg.py' $?
