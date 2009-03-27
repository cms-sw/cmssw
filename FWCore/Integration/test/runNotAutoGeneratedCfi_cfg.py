# Runs using configuration cfi file that contains none of the required
# parameters to test that the required ones are properly inserted
# during validation

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

# expects this cfi file to be in the current directory
# usually run from the tmp/slc4_ia32_gcc345 directory
# where the cfi was just created by the test script
from FWCore.Integration.testProducerWithPsetDescEmpty_cfi import *

process.testProducerWithPsetDescEmpty = testProducerWithPsetDescEmpty

process.p1 = cms.Path(process.testProducerWithPsetDescEmpty)
