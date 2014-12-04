import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.HiTrkEffAnalyzer.hitrkEffAnalyzer_cff import *
hitrkEffAnalyzer_Basic = hitrkEffAnalyzer.clone(
       # evt
       neededCentBins = cms.untracked.vint32(0, 1, 3, 11, 19, 39),
          # simtrk selection
          label_tp_fake = "cutsTPForFak",
          label_tp_effic = "cutsTPForEff",
          # trk selection
          tracks = "hiSelectedTracks",
          useQaulityStr = True,
          qualityString = "highPurity",
         # setup
          fillNtuples = False,
          hasSimInfo = True,
          ptBinScheme = 3, # coarse binning for jet ana
          fiducialCut = False, # whether to put in by hand pixel holes.
          usePxlPair = True, # for acceptance calculation
          # jet-trk
          jets = 'icPu5patJets',
          trkAcceptedJet = True, # jet |eta|<2,
          useJetEtMode = 2, # mode2: jet1, jet2 or jet_pt=0
          jetTrkOnly = False # only trks in 0.8 cone or not
          )

hitrkEffAnalyzer_MergedSelected = hitrkEffAnalyzer_Basic.clone(
       useQaulityStr = False,
          tracks = cms.untracked.InputTag("hiMergedTracksSelcted")
          )

hitrkEffAnalyzer_MergedGeneral = hitrkEffAnalyzer_Basic.clone(
       useQaulityStr = False,
          tracks = cms.untracked.InputTag("hiMergedTracksGeneral")
          )

hitrkEffAnalyzer_MergedGeneralCalo = hitrkEffAnalyzer_Basic.clone(
       useQaulityStr = False,
          tracks = cms.untracked.InputTag("hiMergedTracksGeneralCalo")
          )

hitrkEffAnalyzer_akpu3pf = hitrkEffAnalyzer_Basic.clone(
      jets = 'akPu3PFpatJets',
        )

trackeff_seq = cms.Sequence(
    hitrkEffAnalyzer_MergedSelected *
    hitrkEffAnalyzer_MergedGeneral *
    hitrkEffAnalyzer_MergedGeneralCalo
    )
