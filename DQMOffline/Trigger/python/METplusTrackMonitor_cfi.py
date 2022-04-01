import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.metPlusTrackMonitoring_cfi import metPlusTrackMonitoring
# Define 100 logarithmic bins from 10^0 to 10^3 GeV
binsLogX_METplusTrack = []
nBinsLogX_METplusTrack = 100
powerLo_METplusTrack = 0.0
powerHi_METplusTrack = 3.0
binPowerWidth_METplusTrack = (powerHi_METplusTrack - powerLo_METplusTrack) / nBinsLogX_METplusTrack
for ibin in range(nBinsLogX_METplusTrack + 1):
   binsLogX_METplusTrack.append( pow(10, powerLo_METplusTrack + ibin * binPowerWidth_METplusTrack) )

hltMETplusTrackMonitoring = metPlusTrackMonitoring.clone(
  FolderName = 'HLT/EXO/MET/MET105_IsoTrk50/',
  histoPSet = dict(
          lsPSet = dict(
                    nbins = 250 ,
                    xmin  =  0.,
                    xmax  = 2500.),

          metPSet = dict(
                    nbins = 100,
                    xmin  = -0.5,
                    xmax  = 999.5),

          ptPSet = dict(
                    nbins = 100,
                    xmin  = -0.5,
                    xmax  = 999.5),

          etaPSet = dict(
                    nbins = 24,
                    xmin  = -2.4,
                    xmax  = 2.4),

          phiPSet = dict(
                    nbins = 32,
                    xmin  = -3.2,
                    xmax  = 3.2),
    
          metBinning = binsLogX_METplusTrack,
          ptBinning = binsLogX_METplusTrack
      ),
  met       = "caloMet", # caloMet
  jets      = "ak4PFJets", # ak4PFJets, ak4PFJetsCHS
  muons     = "muons", # while pfIsolatedMuonsEI are reco::PFCandidate !

  muonSelection = 'pt>26 && abs(eta)<2.1 && (pfIsolationR04.sumChargedHadronPt+pfIsolationR04.sumPhotonEt+pfIsolationR04.sumNeutralHadronEt-0.5*pfIsolationR04.sumPUPt)/pt<0.12',
  vtxSelection = 'ndof>=4 && abs(z)<24.0 && position.Rho<2.0',
  nmuons = 1,
  leadJetEtaCut = 2.4,

  numGenericTriggerEventPSet = dict(
    andOr          = False,
    #dbLabel        = "ExoDQMTrigger", # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
    andOrHlt       = True,# True:=OR; False:=AND
    hltInputTag    = "TriggerResults::HLT",
    hltPaths       = ["HLT_MET105_IsoTrk50_v*"], # HLT_ZeroBias_v
    #hltDBKey       = "EXO_HLT_MET",
    errorReplyHlt  = False,
    verbosityLevel = 0
  ),

  denGenericTriggerEventPSet = dict(
    andOr          =  False ,
    dcsInputTag    =  "scalersRawToDigi",
    dcsRecordInputTag = "onlineMetaDataDigis",
    dcsPartitions  = [24, 25, 26, 27, 28, 29], # 24-27: strip, 28-29: pixel, we should add all other detectors !
    andOrDcs       = False,
    errorReplyDcs  = True,
    verbosityLevel = 1,
    hltPaths       = ["HLT_IsoMu27_v*"])
)

