from math import pi

import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.tau3muMonitoring_cfi import tau3muMonitoring

hltTau3Mumonitoring = tau3muMonitoring.clone()

# DQM directory
hltTau3Mumonitoring.FolderName = cms.string('HLT/BPH/Tau3Mu/')

# histogram binning
hltTau3Mumonitoring.histoPSet.ptPSet = cms.PSet(
  nbins = cms.uint32( 40  ),
  xmin  = cms.double(- 0.5),
  xmax  = cms.double( 99.5),
)
hltTau3Mumonitoring.histoPSet.etaPSet = cms.PSet(
  nbins = cms.uint32( 10  ),
  xmin  = cms.double(- 2.6),
  xmax  = cms.double(  2.6),
)
hltTau3Mumonitoring.histoPSet.phiPSet = cms.PSet(
  nbins = cms.uint32( 10),
  xmin  = cms.double(-pi),
  xmax  = cms.double( pi),
)
hltTau3Mumonitoring.histoPSet.massPSet = cms.PSet(
  nbins = cms.uint32( 40  ),
  xmin  = cms.double(  0.5),
  xmax  = cms.double(  3. ),
)

hltTau3Mumonitoring.taus = cms.InputTag("hltTauPt15MuPts711Mass1p3to2p1Iso", "Taus") # 3-muon candidates

hltTau3Mumonitoring.GenericTriggerEventPSet.andOr          = cms.bool( False ) # https://github.com/cms-sw/cmssw/blob/76d343005c33105be1e01b7b7278c07d753398db/CommonTools/TriggerUtils/src/GenericTriggerEventFlag.cc#L249
hltTau3Mumonitoring.GenericTriggerEventPSet.andOrHlt       = cms.bool( True  ) # https://github.com/cms-sw/cmssw/blob/76d343005c33105be1e01b7b7278c07d753398db/CommonTools/TriggerUtils/src/GenericTriggerEventFlag.cc#L114
hltTau3Mumonitoring.GenericTriggerEventPSet.hltPaths       = cms.vstring("HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v*")
# determine the DCS partitions to be active https://github.com/cms-sw/cmssw/blob/b767924e38a6b75340e6e120ece95b648bd11d2d/DataFormats/Scalers/interface/DcsStatus.h#L35
# RPC (12) + DT (13-15) + CSC (16-17) + TRK (24-27) + PIX (28-29)
hltTau3Mumonitoring.GenericTriggerEventPSet.dcsPartitions = cms.vint32(12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28, 29) 
# hltTau3Mumonitoring.GenericTriggerEventPSet.verbosityLevel = cms.uint32(2) # set to 2 for debugging
# hltTau3Mumonitoring.GenericTriggerEventPSet.hltInputTag    = cms.InputTag("TriggerResults::reHLT") # change the process name to reHLT when running tests (if the process used to rerun the HLT is reHLT, of course)

