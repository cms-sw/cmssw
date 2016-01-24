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
      jpar = cms.vdouble(1.31,1.21,1.12,1.25,1.29),
      pjpar = cms.vdouble(-1.4,0.643)

      )
