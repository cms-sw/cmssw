import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.L1TMonitor.L1TStage2uGT_cff import l1tStage2uGMTOutVsuGTIn

# directory path shortening
ugtDqmDir = 'L1T/L1TStage2uGT'

# CaloL2 vs. uGT
l1tStage2uGTvsCaloLayer2RatioClient = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string(ugtDqmDir + '/calol2ouput_vs_uGTinput'),
    inputNum = cms.untracked.string(ugtDqmDir + '/calol2ouput_vs_uGTinput/errorSummaryNum'),
    inputDen = cms.untracked.string(ugtDqmDir + '/calol2ouput_vs_uGTinput/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between CaloLayer2 outputs and uGT inputs'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

# Muons vs. uGT
l1tStage2uGMTOutVsuGTInRatioClient = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string(ugtDqmDir+'/uGMToutput_vs_uGTinput'),
    inputNum = cms.untracked.string(ugtDqmDir+'/uGMToutput_vs_uGTinput/errorSummaryNum'),
    inputDen = cms.untracked.string(ugtDqmDir+'/uGMToutput_vs_uGTinput/errorSummaryDen'),
    ignoreBin = cms.untracked.vint32(l1tStage2uGMTOutVsuGTIn.ignoreBin),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT output muons and uGT input muons'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

# uGT timing
l1tStage2uGTRatioTimingPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(ugtDqmDir+'/'),
    efficiency = cms.vstring(
       "Ratio_First_Bunch_In_Train_Minus_2 'First Bunch In Train Minus 2 Ratio; Bunch crossing number relative to L1A; Algorithm trigger bit' timing_aux/first_bunch_in_train_minus2 timing_aux/den_first_bunch_in_train_minus2",
       "Ratio_First_Bunch_In_Train_Minus_1 'First Bunch In Train Minus 1 Ratio; Bunch crossing number relative to L1A; Algorithm trigger bit' timing_aux/first_bunch_in_train_minus1 timing_aux/den_first_bunch_in_train_minus1",
       "Ratio_First_Bunch_In_Train 'First Bunch In Train Ratio; Bunch crossing number relative to L1A; Algorithm trigger bit' timing_aux/first_bunch_in_train timing_aux/den_first_bunch_in_train",
       "Ratio_Last_Bunch_In_Train 'Last Bunch In Train Ratio; Bunch crossing number relative to L1A; Algorithm trigger bit' timing_aux/last_bunch_in_train timing_aux/den_last_bunch_in_train",
       "Ratio_Isolated_Bunch_In_Train 'Isolated Bunch In Train Ratio; Bunch crossing number relative to L1A; Algorithm trigger bit' timing_aux/isolated_bunch timing_aux/den_isolated_bunch_in_train",
       "Ratio_Prescaled_First_Bunch_In_Train 'First Bunch In Train Ratio for Prescaled Triggers; Bunch crossing number relative to L1A; Algorithm trigger bit' timing_aux/prescaled_algo_first_collision_in_train_ timing_aux/den_prescaled_algo_first_collision_in_train_",
       "Ratio_Prescaled_Isolated_Bunch 'Isolated Bunch Ratio for Prescaled Triggers; Bunch crossing number relative to L1A; Algorithm trigger bit' timing_aux/prescaled_algo_isolated_collision_in_train_ timing_aux/den_prescaled_algo_isolated_collision_in_train_",
       "Ratio_Prescaled_Last_Bunch_In_Train 'Last Bunch In Train Ratio for Prescaled Triggers; Bunch crossing number relative to L1A; Algorithm trigger bit' timing_aux/prescaled_algo_last_collision_in_train_ timing_aux/den_prescaled_algo_last_collision_in_train_",
       "Ratio_Unprescaled_First_Bunch_In_Train 'First Bunch In Train Ratio for Unprescaled Triggers; Bunch crossing number relative to L1A; Algorithm trigger bit' timing_aux/unprescaled_algo_first_collision_in_train_ timing_aux/den_unprescaled_algo_first_collision_in_train_",
       "Ratio_Unprescaled_Isolated_Bunch 'Isolated Bunch Ratio for Unprescaled Triggers; Bunch crossing number relative to L1A; Algorithm trigger bit' timing_aux/unprescaled_algo_isolated_collision_in_train_ timing_aux/den_unprescaled_algo_isolated_collision_in_train_",
       "Ratio_Unprescaled_Last_Bunch_In_Train 'Last Bunch In Train Ratio for Unprescaled Triggers; Bunch crossing number relative to L1A; Algorithm trigger bit' timing_aux/unprescaled_algo_last_collision_in_train_ timing_aux/den_unprescaled_algo_last_collision_in_train_",
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(0),
    runOnEndLumi = cms.untracked.bool(True),
    makeGlobalEffienciesPlot = cms.untracked.bool(False)
)

# sequences
l1tStage2uGTClient = cms.Sequence(
    l1tStage2uGTvsCaloLayer2RatioClient +
    l1tStage2uGMTOutVsuGTInRatioClient +
    l1tStage2uGTRatioTimingPlots
)
