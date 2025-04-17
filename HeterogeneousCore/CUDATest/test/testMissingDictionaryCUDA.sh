#! /bin/bash

function die {
  echo -e "$1"
  exit 1
}

LOCAL_TEST_DIR=${SCRAM_TEST_PATH:-$CMSSW_BASE/src/HeterogeneousCore/CUDATest/test}

cmsRun ${LOCAL_TEST_DIR}/testMissingDictionaryCUDA_cfg.py >& testMissingDictionaryCUDA.log && die "The cmsRun test job succeeded unexpectedly"
grep -q "An exception of category 'DictionaryNotFound' occurred" testMissingDictionaryCUDA.log || die "Cannot find the following string in the exception message:\nAn exception of category 'DictionaryNotFound' occurred"
grep -q "edm::Wrapper<edmtest::MissingDictionaryCUDAObject>"     testMissingDictionaryCUDA.log || die "Cannot find the following string in the exception message:\nedm::Wrapper<edmtest::MissingDictionaryCUDAObject>"
