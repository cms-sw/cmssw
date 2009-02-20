import FWCore.ParameterSet.Config as cms

hfreco = cms.EDFilter("HcalHitReconstructor",
                      correctionPhaseNS = cms.double(0.0),
                      digiLabel = cms.InputTag("hcalDigis"),
                      samplesToAdd = cms.int32(1),
                      Subdetector = cms.string('HF'),
                      firstSample = cms.int32(3),
                      correctForPhaseContainment = cms.bool(False),
                      correctForTimeslew = cms.bool(False),
                      
                      # Tags for calculating status flags
                      digistat= cms.PSet(
                        HFpulsetimemin     = cms.int32(0),
                        HFpulsetimemax     = cms.int32(10), # min/max time slice values for peak
                        HFratio_beforepeak = cms.double(0.1), # max allowed ratio
                        HFratio_afterpeak  = cms.double(1.0), # max allowed ratio
                      ),
                      rechitstat=cms.PSet(
                        HFlongshortratio = cms.double(0.99), # max allowed ratio of (L-S)/(L+S)
                      )
                  )


