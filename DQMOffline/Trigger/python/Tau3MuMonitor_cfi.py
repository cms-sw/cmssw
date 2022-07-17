from math import pi

import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.tau3muMonitoring_cfi import tau3muMonitoring

hltTau3Mumonitoring = tau3muMonitoring.clone(

  # DQM directory
  FolderName = 'HLT/BPH/Tau3Mu/',

  # histogram binning
  histoPSet = dict(
      ptPSet = dict(
          nbins =  40  ,
          xmin  = - 0.5,
          xmax  =  99.5),

      etaPSet = dict(
          nbins = 10  ,
          xmin  = - 2.6,
          xmax  =  2.6),

      phiPSet = dict(
          nbins = 10,
          xmin  = -pi,
          xmax  =  pi),

      massPSet = dict(
          nbins =  40  ,
          xmin  =  0.5,
          xmax  =  3. ),
  ),

  taus = "hltTauPt15MuPts711Mass1p3to2p1Iso:Taus", # 3-muon candidates

  GenericTriggerEventPSet = dict(
    andOr          = False, # https://github.com/cms-sw/cmssw/blob/76d343005c33105be1e01b7b7278c07d753398db/CommonTools/TriggerUtils/src/GenericTriggerEventFlag.cc#L249
    andOrHlt       =  True , # https://github.com/cms-sw/cmssw/blob/76d343005c33105be1e01b7b7278c07d753398db/CommonTools/TriggerUtils/src/GenericTriggerEventFlag.cc#L114
    hltPaths       = ["HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v*"],
    # determine the DCS partitions to be active https://github.com/cms-sw/cmssw/blob/b767924e38a6b75340e6e120ece95b648bd11d2d/DataFormats/Scalers/interface/DcsStatus.h#L35
    # RPC (12) + DT (13-15) + CSC (16-17) + TRK (24-27) + PIX (28-29)
    dcsPartitions = [12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28, 29])
    # verbosityLevel = 2, # set to 2 for debugging
    # hltInputTag    = "TriggerResults::reHLT") # change the process name to reHLT when running tests (if the process used to rerun the HLT is reHLT, of course)
  )
