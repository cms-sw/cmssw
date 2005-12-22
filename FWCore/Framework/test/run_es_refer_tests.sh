#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

(cmsRun test_es_prefer_prods.cfg ) || die 'Failure using test_es_prefer_prods.cfg' $?
(cmsRun test_es_prefer_sources.cfg ) || die 'Failure using test_es_prefer_sources.cfg' $?
(cmsRun test_es_prefer_source_beats_prod.cfg ) || die 'Failure using test_es_prefer_source_beats_prod.cfg' $?
(cmsRun test_prod_trumps_source.cfg ) || die 'Failure using test_prod_trumps_source.cfg' $?


