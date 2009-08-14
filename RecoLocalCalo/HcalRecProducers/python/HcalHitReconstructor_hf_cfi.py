import FWCore.ParameterSet.Config as cms

hfreco = cms.EDFilter("HcalHitReconstructor",
                      correctionPhaseNS = cms.double(0.0),
                      digiLabel = cms.InputTag("hcalDigis"),
                      samplesToAdd = cms.int32(1),
                      Subdetector = cms.string('HF'),
                      firstSample = cms.int32(3),
                      correctForPhaseContainment = cms.bool(False),
                      correctForTimeslew = cms.bool(False),
                      dropZSmarkedPassed = cms.bool(True),

                      # Tags for calculating status flags
                      correctTiming = cms.bool(True),
                      setNoiseFlags = cms.bool(True),
                      setHSCPFlags  = cms.bool(True),
                      setSaturationFlags = cms.bool(True),
                      setTimingTrustFlags = cms.bool(True),
                      
                      digistat= cms.PSet(
                        HFpulsetimemin     = cms.int32(0),
                        HFpulsetimemax     = cms.int32(10), # min/max time slice values for peak
                        HFratio_beforepeak = cms.double(0.6), # max allowed ratio (started at 0.1, loosened to 0.6 after pion studies)
                        HFratio_afterpeak  = cms.double(1.0), # max allowed ratio
                        HFadcthreshold       = cms.int32(10), # minimum size of peak (in ADC counts, after ped subtraction) to be considered noisy
                      ),
                      rechitstat=cms.PSet(
                        HFlongshortratio = cms.double(0.99), # max allowed ratio of (L-S)/(L+S)
                        HFthresholdET = cms.double(2.0), # minimum energy (in GeV) required for a cell to be considered hot (started at 0.5, loosened to 2.0 after pion studies)
                      ),
                      saturationParameters=  cms.PSet(maxADCvalue=cms.int32(127)),
                      hfTimingTrustParameters = cms.PSet(
                        hfTimingTrustLevel1=cms.int32(1), # 1ns timing accuracy
                        hfTimingTrustLevel2=cms.int32(4)  # 4ns timing accuracy
                      )
                        
                  )


