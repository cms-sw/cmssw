import FWCore.ParameterSet.Config as cms

METSignificanceParams = cms.PSet(

      # jet resolutions
      jetThreshold = cms.double(15),
      
      #jet-lepton matching dR
      dRMatch = cms.double(0.4),

      # eta bins for jet resolution tuning
      jeta = cms.vdouble(0.8, 1.3, 1.9, 2.5),

      # tuning parameters
      #Run I, based on 53X / JME-13-003
      #jpar = cms.vdouble(1.20,1.13,1.03,0.96,1.08),
      #pjpar = cms.vdouble(-1.9,0.6383)
      #Run II MC, based on 80X
      jpar = cms.vdouble(1.39,1.26,1.21,1.23,1.28),
      pjpar = cms.vdouble(-0.2586,0.6173),
      )

METSignificanceParams_Data=cms.PSet(

      # jet resolutions
      jetThreshold = cms.double(15),
      
      #jet-lepton matching dR
      dRMatch = cms.double(0.4),

      # eta bins for jet resolution tuning
      jeta = cms.vdouble(0.8, 1.3, 1.9, 2.5),

      # tuning parameters
      #Run I, based on 53X / JME-13-003
      #jpar = cms.vdouble(1.20,1.13,1.03,0.96,1.08),
      #pjpar = cms.vdouble(-1.9,0.6383)
      #Run II data, based on 80X
      jpar = cms.vdouble(1.38,1.28,1.22,1.16,1.10),
      pjpar = cms.vdouble(0.0033,0.5802),
      )
