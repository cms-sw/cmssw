import FWCore.ParameterSet.Config as cms

METSignificanceParams = cms.PSet(

      # jet resolutions
      phiResFile   = cms.string('Spring10_PhiResolution_AK5PF.txt'),
      jetThreshold = cms.double(15),
      
      #jet-lepton matching dR
      dRMatch = cms.double(0.4),

      # eta bins for jet resolution tuning
      jeta = cms.vdouble(0.5, 1.1, 1.7, 2.3),

      # tuning parameters
      jpar = cms.vdouble(1.24,1.12,1.01,1.11,1.03),
      pjpar = cms.vdouble(-2.0,0.6394)

      )
