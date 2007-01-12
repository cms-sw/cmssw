#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

(cmsRun ${LOCAL_TEST_DIR}/test_es_prefer_prods.cfg ) || die 'Failure using test_es_prefer_prods.cfg' $?
(cmsRun ${LOCAL_TEST_DIR}/test_es_prefer_sources.cfg ) || die 'Failure using test_es_prefer_sources.cfg' $?
(cmsRun ${LOCAL_TEST_DIR}/test_es_prefer_source_beats_prod.cfg ) || die 'Failure using test_es_prefer_source_beats_prod.cfg' $?
(cmsRun ${LOCAL_TEST_DIR}/test_prod_trumps_source.cfg ) || die 'Failure using test_prod_trumps_source.cfg' $?
(cmsRun ${LOCAL_TEST_DIR}/test_es_prefer_2_es_sources_order1.cfg ) || die 'Failure using test_es_prefer_2_es_sources_order1.cfg' $?
(cmsRun ${LOCAL_TEST_DIR}/test_es_prefer_2_es_sources_order2.cfg ) || die 'Failure using test_es_prefer_2_es_sources_order2.cfg' $?
!(cmsRun ${LOCAL_TEST_DIR}/test_2_es_sources_no_prefer.cfg ) || die 'Should have failed using test_2_es_sources_no_prefer.cfg' $?
