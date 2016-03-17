import FWCore.ParameterSet.Config as cms

METSignificanceParams = cms.PSet(

      # jet resolutions
      jetThreshold = cms.double(15),
      
      #jet-lepton matching dR
      dRMatch = cms.double(0.4),

      # eta bins for jet resolution tuning
      jeta = cms.vdouble(0.8, 1.3, 1.9, 2.5),

      # tuning parameters
      jpar = cms.vdouble(1.20,1.13,1.03,0.96,1.08),
      pjpar = cms.vdouble(-1.9,0.6383)

      )
