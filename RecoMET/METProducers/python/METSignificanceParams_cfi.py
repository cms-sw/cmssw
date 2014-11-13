import FWCore.ParameterSet.Config as cms

METSignificanceParams = cms.PSet(

      # jet resolutions
      jetResAlgo    = cms.string('AK5PF'),
      jetResEra     = cms.string('Spring10'),
      jetThreshold  = cms.double(20),

      # eta bins for jet resolution tuning
      jeta = cms.vdouble(0.5, 1.1, 1.7, 2.3),

      # tuning parameters
      jpar = cms.vdouble(1.15061, 1.07776, 1.04204, 1.12509, 1.56414, 0.0, 0.548758)

      )
