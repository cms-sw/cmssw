import FWCore.ParameterSet.Config as cms

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
options = cms.untracked.PSet(
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)
