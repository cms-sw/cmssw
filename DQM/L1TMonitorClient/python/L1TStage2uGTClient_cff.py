import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.L1TMonitor.L1TStage2uGT_cff import l1tStage2uGMTOutVsuGTIn

# directory path shortening
ugtDqmDir = 'L1T/L1TStage2uGT'
ugtBoardCompDqmDir = ugtDqmDir+'/uGTBoardComparisons'

# input histograms
errHistNumStr = 'errorSummaryNum'
errHistDenStr = 'errorSummaryDen'

# CaloL2 vs. uGT
l1tStage2uGTvsCaloLayer2RatioClient = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string(ugtDqmDir + '/calol2ouput_vs_uGTinput'),
    inputNum = cms.untracked.string(ugtDqmDir + '/calol2ouput_vs_uGTinput/'+errHistNumStr),
    inputDen = cms.untracked.string(ugtDqmDir + '/calol2ouput_vs_uGTinput/'+errHistDenStr),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between CaloLayer2 outputs and uGT inputs'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

# Muons vs. uGT
l1tStage2uGMTOutVsuGTInRatioClient = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string(ugtDqmDir+'/uGMToutput_vs_uGTinput'),
    inputNum = cms.untracked.string(ugtDqmDir+'/uGMToutput_vs_uGTinput/'+errHistNumStr),
    inputDen = cms.untracked.string(ugtDqmDir+'/uGMToutput_vs_uGTinput/'+errHistDenStr),
    ignoreBin = cms.untracked.vint32(l1tStage2uGMTOutVsuGTIn.ignoreBin),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT output muons and uGT input muons'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

## uGT Board Comparisons

l1tStage2uGTMuon1vsMuon2RatioClient = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard2/Muons'),
    inputNum = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard2/Muons/'+errHistNumStr),
    inputDen = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard2/Muons/'+errHistDenStr),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of Mismatch Rates between Muons from uGT Board 1 and uGT Board 2'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

l1tStage2uGTMuon1vsMuon3RatioClient = l1tStage2uGTMuon1vsMuon2RatioClient.clone() 
l1tStage2uGTMuon1vsMuon3RatioClient.monitorDir = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard3/Muons')
l1tStage2uGTMuon1vsMuon3RatioClient.inputNum = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard3/Muons/'+errHistNumStr)
l1tStage2uGTMuon1vsMuon3RatioClient.inputDen = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard3/Muons/'+errHistDenStr)
l1tStage2uGTMuon1vsMuon3RatioClient.ratioTitle = cms.untracked.string('Summary of Mismatch Rates between Muons from uGT Board 1 and uGT Board 3')

l1tStage2uGTMuon1vsMuon4RatioClient = l1tStage2uGTMuon1vsMuon2RatioClient.clone() 
l1tStage2uGTMuon1vsMuon4RatioClient.monitorDir = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard4/Muons')
l1tStage2uGTMuon1vsMuon4RatioClient.inputNum = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard4/Muons/'+errHistNumStr)
l1tStage2uGTMuon1vsMuon4RatioClient.inputDen = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard4/Muons/'+errHistDenStr)
l1tStage2uGTMuon1vsMuon4RatioClient.ratioTitle = cms.untracked.string('Summary of Mismatch Rates between Muons from uGT Board 1 and uGT Board 4')

l1tStage2uGTMuon1vsMuon5RatioClient = l1tStage2uGTMuon1vsMuon2RatioClient.clone() 
l1tStage2uGTMuon1vsMuon5RatioClient.monitorDir = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard5/Muons')
l1tStage2uGTMuon1vsMuon5RatioClient.inputNum = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard5/Muons/'+errHistNumStr)
l1tStage2uGTMuon1vsMuon5RatioClient.inputDen = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard5/Muons/'+errHistDenStr)
l1tStage2uGTMuon1vsMuon5RatioClient.ratioTitle = cms.untracked.string('Summary of Mismatch Rates between Muons from uGT Board 1 and uGT Board 5')

l1tStage2uGTMuon1vsMuon6RatioClient = l1tStage2uGTMuon1vsMuon2RatioClient.clone() 
l1tStage2uGTMuon1vsMuon6RatioClient.monitorDir = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard6/Muons')
l1tStage2uGTMuon1vsMuon6RatioClient.inputNum = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard6/Muons/'+errHistNumStr)
l1tStage2uGTMuon1vsMuon6RatioClient.inputDen = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard6/Muons/'+errHistDenStr)
l1tStage2uGTMuon1vsMuon6RatioClient.ratioTitle = cms.untracked.string('Summary of Mismatch Rates between Muons from uGT Board 1 and uGT Board 6')

l1tStage2uGTBoardCompMuonsRatioClientSeq = cms.Sequence(
    l1tStage2uGTMuon1vsMuon2RatioClient +
    l1tStage2uGTMuon1vsMuon3RatioClient +
    l1tStage2uGTMuon1vsMuon4RatioClient +
    l1tStage2uGTMuon1vsMuon5RatioClient +
    l1tStage2uGTMuon1vsMuon6RatioClient
)

l1tStage2uGTCalo1vsCalo2RatioClient = l1tStage2uGTMuon1vsMuon2RatioClient.clone() 
l1tStage2uGTCalo1vsCalo2RatioClient.monitorDir = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard2/CaloLayer2')
l1tStage2uGTCalo1vsCalo2RatioClient.inputNum = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard2/CaloLayer2/'+errHistNumStr)
l1tStage2uGTCalo1vsCalo2RatioClient.inputDen = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard2/CaloLayer2/'+errHistDenStr)
l1tStage2uGTCalo1vsCalo2RatioClient.ratioTitle = cms.untracked.string('Summary of Mismatch Rates between CaloLayer2 Inputs from uGT Board 1 and uGT Board 2')

l1tStage2uGTCalo1vsCalo3RatioClient = l1tStage2uGTMuon1vsMuon2RatioClient.clone() 
l1tStage2uGTCalo1vsCalo3RatioClient.monitorDir = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard3/CaloLayer2')
l1tStage2uGTCalo1vsCalo3RatioClient.inputNum = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard3/CaloLayer2/'+errHistNumStr)
l1tStage2uGTCalo1vsCalo3RatioClient.inputDen = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard3/CaloLayer2/'+errHistDenStr)
l1tStage2uGTCalo1vsCalo3RatioClient.ratioTitle = cms.untracked.string('Summary of Mismatch Rates between CaloLayer2 Inputs from uGT Board 1 and uGT Board 3')

l1tStage2uGTCalo1vsCalo4RatioClient = l1tStage2uGTMuon1vsMuon2RatioClient.clone() 
l1tStage2uGTCalo1vsCalo4RatioClient.monitorDir = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard4/CaloLayer2')
l1tStage2uGTCalo1vsCalo4RatioClient.inputNum = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard4/CaloLayer2/'+errHistNumStr)
l1tStage2uGTCalo1vsCalo4RatioClient.inputDen = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard4/CaloLayer2/'+errHistDenStr)
l1tStage2uGTCalo1vsCalo4RatioClient.ratioTitle = cms.untracked.string('Summary of Mismatch Rates between CaloLayer2 Inputs from uGT Board 1 and uGT Board 4')

l1tStage2uGTCalo1vsCalo5RatioClient = l1tStage2uGTMuon1vsMuon2RatioClient.clone() 
l1tStage2uGTCalo1vsCalo5RatioClient.monitorDir = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard5/CaloLayer2')
l1tStage2uGTCalo1vsCalo5RatioClient.inputNum = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard5/CaloLayer2/'+errHistNumStr)
l1tStage2uGTCalo1vsCalo5RatioClient.inputDen = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard5/CaloLayer2/'+errHistDenStr)
l1tStage2uGTCalo1vsCalo5RatioClient.ratioTitle = cms.untracked.string('Summary of Mismatch Rates between CaloLayer2 Inputs from uGT Board 1 and uGT Board 5')

l1tStage2uGTCalo1vsCalo6RatioClient = l1tStage2uGTMuon1vsMuon2RatioClient.clone() 
l1tStage2uGTCalo1vsCalo6RatioClient.monitorDir = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard6/CaloLayer2')
l1tStage2uGTCalo1vsCalo6RatioClient.inputNum = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard6/CaloLayer2/'+errHistNumStr)
l1tStage2uGTCalo1vsCalo6RatioClient.inputDen = cms.untracked.string(ugtBoardCompDqmDir+'/Board1vsBoard6/CaloLayer2/'+errHistDenStr)
l1tStage2uGTCalo1vsCalo6RatioClient.ratioTitle = cms.untracked.string('Summary of Mismatch Rates between CaloLayer2 Inputs from uGT Board 1 and uGT Board 6')

l1tStage2uGTBoardCompCaloLayer2RatioClientSeq = cms.Sequence(
    l1tStage2uGTCalo1vsCalo2RatioClient +
    l1tStage2uGTCalo1vsCalo3RatioClient +
    l1tStage2uGTCalo1vsCalo4RatioClient +
    l1tStage2uGTCalo1vsCalo5RatioClient +
    l1tStage2uGTCalo1vsCalo6RatioClient
)

l1tStage2uGTBoardCompRatioClientSeq = cms.Sequence(
    l1tStage2uGTBoardCompMuonsRatioClientSeq +
    l1tStage2uGTBoardCompCaloLayer2RatioClientSeq
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
    l1tStage2uGTBoardCompRatioClientSeq + 
    l1tStage2uGTRatioTimingPlots
)
