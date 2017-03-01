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
      #Run II MC, based on 76X
      #https://indico.cern.ch/event/527789/contributions/2160488/attachments/1271716/1884792/nmirman_20160511.pdf
      jpar = cms.vdouble(1.29,1.19,1.07,1.13,1.12),
      pjpar = cms.vdouble(-0.04,0.6504),
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
      #Run II data, based on 76X
      #https://indico.cern.ch/event/527789/contributions/2160488/attachments/1271716/1884792/nmirman_20160511.pdf
      jpar = cms.vdouble(1.26,1.14,1.13,1.13,1.06),
      pjpar = cms.vdouble(-3.3,0.5961),
      )
