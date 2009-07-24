import FWCore.ParameterSet.Config as cms

#from DQM.HLTEvF.HLTMonElectron_cfi import *
#from DQM.HLTEvF.electronDQMIsoDistTrigger_cfi import *
#from DQM.HLTEvF.electronDQMIsoDist_cfi import *
#from DQM.HLTEvF.electronDQMPixelMatch_cfi import *
#from DQM.HLTEvF.electronDQMConsumer_cfi import *
#hltMonElectronPath = cms.Path(hltMonE)
#hltMonElectronPath = cms.Path(hltMonE*electronDQMIsoDistTrigger*electronDQMIsoDist*electronDQMPixelMatch*electronDQMConsumer)


from DQM.HLTEvF.electron_test_cff import *
hltMonElectronPath = cms.Path(sourcePlots)
