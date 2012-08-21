import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1TOccupancyTestParameters_cff import *

l1tOccupancyClient = cms.EDAnalyzer("L1TOccupancyClient",
  verbose = cms.bool(False),
  testParams = cms.VPSet(
    #---------------------------------------------
    # Example configuration
    #---------------------------------------------
    #cms.PSet(
      #testName       = cms.string('TauJetsOccEtaPhi'), #test name
      #algoName       = cms.string('XYSymmetry'),       #test to be performed
      #algoType       = cms.int32(1),                   #0=qTest, 1=intrinsic
      #algoParams     = cms.PSet(
        #currDir      = cms.string('L1T/L1TGCT'),       #dir of histogram to be inspected
        #histo        = cms.string('TauJetsOccEtaPhi'), #histogram to be inspected
        #rebinFactorX = cms.int32(1),                   # Rebin factor (X Axis) to apply over tested histogram
        #rebinFactorY = cms.int32(1),                   # Rebin factor (Y Axis) to apply over tested histogram
        #axis         = cms.int32(1),                   # Symetry axis is 1=vertical, 2=horizontal
        #takeCenter   = cms.bool(True),                 # Take central bin (odd bins) or first bin to the right of center (even bins) as axisSymmetryValue
        #axisSymmetryValue      = cms.double(10.5),     #symmetry point on axis to be checked, neglected if takeCenter=true
        #averageMode = cms.int32(2),                    #reference formula for average (0=testCase, 1=arithmetic average, 2=median)
        #factorlow       = cms.double(0.1),             #factor f : mu1=f*mu0
        #factorup        = cms.double(2.0),             #factor f : mu1=f*mu0
        #params_mu0_low  = mu0_x0p1,                    #parameters for determination of mu0_min
        #params_mu0_up   = mu0_x2,                      #parameters for determination of mu0_min
        #params_chi2_low = chi2_x0p1,                   #parameters for determination of chi2-threshold
        #params_chi2_up  = chi2_x2,                     #parameters for determination of chi2-threshold
        #markers         = cms.VPSet(
          #0=Histogram Units, 1=Bin Units (starting with 1,1)
          #cms.PSet(kind=cms.int32(2),xmin=cms.double(1), xmax=cms.double(4), ymin=cms.double(1),ymax=cms.double(18)),
          #cms.PSet(kind=cms.int32(2),xmin=cms.double(19),xmax=cms.double(22),ymin=cms.double(1),ymax=cms.double(18))
        #)
      #)
    #),
    cms.PSet(
      testName   = cms.string('dttf_01_tracks_occupancy_test_summary'), #test name
      algoParams = cms.PSet(
        histPath    = cms.string('L1T/L1TDTTF/09-TEST/dttf_01_tracks_occupancy_test_summary'),
        maskedAreas = cms.VPSet(
          cms.PSet(kind=cms.int32(1),xmin=cms.double(3) ,xmax=cms.double(3) ,ymin=cms.double(1),ymax=cms.double(12)),
        )
      )
    ),
    cms.PSet(
      testName   = cms.string('TauJetsOccEtaPhi'), #test name
      algoParams = cms.PSet(
        histPath        = cms.string('L1T/L1TGCT/TauJetsOccEtaPhi'),
        maskedAreas = cms.VPSet(
          cms.PSet(kind=cms.int32(1),xmin=cms.double(1) ,xmax=cms.double(4) ,ymin=cms.double(1),ymax=cms.double(18)),
          cms.PSet(kind=cms.int32(1),xmin=cms.double(19),xmax=cms.double(22),ymin=cms.double(1),ymax=cms.double(18))
        )
      )
    ),
    cms.PSet(
      testName   = cms.string('AllJetsOccEtaPhi'), #test name
      algoParams = cms.PSet(
        histPath        = cms.string('L1T/L1TGCT/AllJetsOccEtaPhi'),
        maskedAreas = cms.VPSet()
      )
    ),
    cms.PSet(
      testName   = cms.string('RctRegionsOccEtaPhi'), #test name
      algoParams = cms.PSet(
        histPath        = cms.string('L1T/L1TRCT/RctRegionsOccEtaPhi'),
        maskedAreas = cms.VPSet()
      )
    ),
    cms.PSet(
      testName   = cms.string('RctEmIsoEmOccEtaPhi'), #test name
      algoParams = cms.PSet(
        histPath        = cms.string('L1T/L1TRCT/RctEmIsoEmOccEtaPhi'),
        maskedAreas = cms.VPSet(
          cms.PSet(kind=cms.int32(1),xmin=cms.double(1), xmax=cms.double(4), ymin=cms.double(1),ymax=cms.double(18)),
          cms.PSet(kind=cms.int32(1),xmin=cms.double(19),xmax=cms.double(22),ymin=cms.double(1),ymax=cms.double(18))
        )
      )
    ),
    cms.PSet(
      testName   = cms.string('RctEmNonIsoEmOccEtaPhi'), #test name
      algoParams = cms.PSet(
        histPath        = cms.string('L1T/L1TRCT/RctEmNonIsoEmOccEtaPhi'),
        maskedAreas = cms.VPSet(
          cms.PSet(kind=cms.int32(1),xmin=cms.double(1), xmax=cms.double(4), ymin=cms.double(1),ymax=cms.double(18)),
          cms.PSet(kind=cms.int32(1),xmin=cms.double(19),xmax=cms.double(22),ymin=cms.double(1),ymax=cms.double(18))
        )
      )
    ),
    cms.PSet(
      testName   = cms.string('NonIsoEmOccEtaPhi'), #test name
      algoParams = cms.PSet(
        histPath        = cms.string('L1T/L1TGCT/NonIsoEmOccEtaPhi'),
        maskedAreas = cms.VPSet(
          cms.PSet(kind=cms.int32(1),xmin=cms.double(1), xmax=cms.double(4), ymin=cms.double(1),ymax=cms.double(18)),
          cms.PSet(kind=cms.int32(1),xmin=cms.double(19),xmax=cms.double(22),ymin=cms.double(1),ymax=cms.double(18))
        )
      )
    ),
    cms.PSet(
      testName   = cms.string('IsoEmOccEtaPhi'), #test name
      algoParams = cms.PSet(
        histPath        = cms.string('L1T/L1TGCT/IsoEmOccEtaPhi'),
        maskedAreas = cms.VPSet(
          cms.PSet(kind=cms.int32(1),xmin=cms.double(1), xmax=cms.double(4), ymin=cms.double(1),ymax=cms.double(18)),
          cms.PSet(kind=cms.int32(1),xmin=cms.double(19),xmax=cms.double(22),ymin=cms.double(1),ymax=cms.double(18))
        )
      )
    )

    #----------------------------------------------------
    # Other tests that may be activated in the future
    #----------------------------------------------------
    #cms.PSet(
      #testName      = cms.string('GMT_etaphi'), #test name
      #algoParams    = cms.PSet(
        #currDir     = cms.string('L1T/L1TGMT/GMT_etaphi'), #dir of histogram to be inspected
        #maskedAreas = cms.VPSet()
      #)
    #),
    #cms.PSet(
      #testName       = cms.string('RPCTF_muons_eta_phi_bx0'), #test name
      #algoParams     = cms.PSet(
        #currDir      = cms.string('L1T/L1TRPCTF/RPCTF_muons_eta_phi_bx0'),
        #rebinFactorX = cms.int32(1), # Rebin factor (X Axis) to apply over tested histogram
        #rebinFactorY = cms.int32(12),# Rebin factor (Y Axis) to apply over tested histogram
        #maskedAreas  = cms.VPSet(
          #1...units of histogram, 2...x-/y-internal coordinates (starting with 1,1) (rebinned!)
          #cms.PSet(kind=cms.int32(1),xmin=cms.double(1), xmax=cms.double(4), ymin=cms.double(1),ymax=cms.double(12)),
          #cms.PSet(kind=cms.int32(1),xmin=cms.double(30),xmax=cms.double(33),ymin=cms.double(1),ymax=cms.double(12))
        #)
      #)
    #),
    #cms.PSet(
      #testName      = cms.string('CSCTF_Chamber_Occupancies'), #test name
      #algoParams    = cms.PSet(
        #currDir     = cms.string('L1T/L1TCSCTF/CSCTF_Chamber_Occupancies'),
        #axis        = cms.int32(2), #which axis should be checked? 1=eta, 2=phi
        #maskedAreas = cms.VPSet(
          #1...units of histogram, 2...x-/y-internal coordinates (starting with 1,1)
          #cms.PSet(kind=cms.int32(1),xmin=cms.double(4), xmax=cms.double(9), ymin=cms.double(1), ymax=cms.double(1)),
          #cms.PSet(kind=cms.int32(1),xmin=cms.double(4), xmax=cms.double(9), ymin=cms.double(10),ymax=cms.double(10)),
          #cms.PSet(kind=cms.int32(1),xmin=cms.double(22),xmax=cms.double(27),ymin=cms.double(1), ymax=cms.double(1)),
          #cms.PSet(kind=cms.int32(1),xmin=cms.double(22),xmax=cms.double(27),ymin=cms.double(10),ymax=cms.double(10)),
          #cms.PSet(kind=cms.int32(1),xmin=cms.double(40),xmax=cms.double(45),ymin=cms.double(1), ymax=cms.double(1)),
          #cms.PSet(kind=cms.int32(1),xmin=cms.double(40),xmax=cms.double(45),ymin=cms.double(10),ymax=cms.double(10)),
          #cms.PSet(kind=cms.int32(1),xmin=cms.double(49),xmax=cms.double(54),ymin=cms.double(1), ymax=cms.double(1)),
          #cms.PSet(kind=cms.int32(1),xmin=cms.double(49),xmax=cms.double(54),ymin=cms.double(10),ymax=cms.double(10))
        #)
      #)
    #),
    #cms.PSet(
      #testName      = cms.string('dttf_03_tracks_occupancy_summary'), #test name
      #algoParams    = cms.PSet(
        #currDir     = cms.string('L1T/L1TDTTF/01-INCLUSIVE/dttf_03_tracks_occupancy_summary'),
        #maskedAreas = cms.VPSet()
      #)
    #),
    #cms.PSet(
      #testName      = cms.string('dttf_12_phi_vs_eta'), #test name
      #algoParams    = cms.PSet(
        #currDir     = cms.string('L1T/L1TDTTF/01-INCLUSIVE/dttf_12_phi_vs_eta'),
        #maskedAreas = cms.VPSet()
      #)
    #)
  )
)
