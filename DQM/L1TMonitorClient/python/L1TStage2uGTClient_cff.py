import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

# directory path shortening
ugtDqmDir = 'L1T/L1TStage2uGT'
# input histograms
errHistNumStr = 'errorSummaryNum'
errHistDenStr = 'errorSummaryDen'

# Muons
l1tStage2uGMTOutVsuGTInRatioClient = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string(ugtDqmDir+'/uGMToutput_vs_uGTinput'),
    inputNum = cms.untracked.string(ugtDqmDir+'/uGMToutput_vs_uGTinput/'+errHistNumStr),
    inputDen = cms.untracked.string(ugtDqmDir+'/uGMToutput_vs_uGTinput/'+errHistDenStr),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT output muons and uGT input muons'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)


dqmRatioTimingPlots = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("L1T/L1TStage2uGT/"),
    monitorDir = cms.untracked.string(ugtDqmDir),
    efficiency = cms.vstring(
       "Ratio_First_Bunch_In_Train 'First Bunch In Train Ratio; Bunch crossing number relative to L1A; Algorithm trigger bit' first_bunch_in_train den_first_bunch_in_train",
       "Ratio_Last_Bunch_In_Train 'Last Bunch In Train Ratio; Bunch crossing number relative to L1A; Algorithm trigger bit' last_bunch_in_train den_last_bunch_in_train",
       "Ratio_Isolated_Bunch_In_Train 'Isolated Bunch In Train Ratio; Bunch crossing number relative to L1A; Algorithm trigger bit' isolated_bunch den_isolated_bunch_in_train",
    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string(""),
    verbose = cms.untracked.uint32(0),
)
    

# sequences
l1tStage2uGTClient = cms.Sequence(
    l1tStage2uGMTOutVsuGTInRatioClient +
    dqmRatioTimingPlots 
)
