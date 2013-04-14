# The following comments couldn't be translated into the new config version:

# Test storing OtherThing as well
# Configuration file for PrePoolInputTest 

import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTBOTHFILES")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.OtherThing = cms.EDProducer("OtherThingProducer")

process.source = cms.Source("PoolSource",
                            secondaryFileNames = cms.untracked.vstring("file:PoolInputOther.root"),
                            fileNames = cms.untracked.vstring("file:PoolInput2FileTest.root")
                            )


process.p = cms.Path(process.OtherThing)



