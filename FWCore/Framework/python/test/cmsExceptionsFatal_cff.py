import FWCore.ParameterSet.Config as cms

import FWCore.Framework.test.cmsExceptionsFatalOption_cff

options = cms.untracked.PSet(
    Rethrow=FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)
#test
y =10
x=2
print(x)
