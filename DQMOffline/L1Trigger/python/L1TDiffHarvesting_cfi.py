import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tDiffHarvesting = DQMEDHarvester(
    "L1TDiffHarvesting",
    verbose=cms.untracked.bool(False),
    plotCfgs=cms.untracked.VPSet(
        cms.untracked.PSet(
            cms.untracked.PSet(  # EMU comparison
                dir1=cms.untracked.string(
                    "L1T/L1TStage2CaloLayer2"),
                dir2=cms.untracked.string(
                    "L1TEMU/L1TStage2CaloLayer2"),
                outputDir=cms.untracked.string(
                    "L1TEMU/L1TStage2CaloLayer2/Comparison"),
                plots=cms.untracked.vstring(
                    "resolutionJetET_HB", "resolutionJetET_HE", "resolutionJetET_HF",
                    "resolutionJetET_HB_HE", "resolutionJetPhi_HB", "resolutionJetPhi_HE",
                    "resolutionJetPhi_HF", "resolutionJetPhi_HB_HE", "resolutionJetEta",
                    "resolutionMET", "resolutionMHT", "resolutionETT", "resolutionHTT",
                    "resolutionMETPhi", "resolutionMHTPhi",
                )
            )
        )
    )

)
