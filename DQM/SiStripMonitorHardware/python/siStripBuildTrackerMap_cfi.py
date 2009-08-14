import FWCore.ParameterSet.Config as cms

siStripBuildTrackerMap = cms.EDAnalyzer(
    "BuildTrackerMapPlugin",
    #input root file containing histograms
    InputFileName = cms.untracked.string('DQMStore.root'),
    #name of tkHistoMap to dump
    TkHistoMapNameVec = cms.untracked.vstring('TkHMap_MeanCMAPV0','TkHMap_MeanCMAPV1','TkHMap_RmsCMAPV0','TkHMap_RmsCMAPV1'),
    #Name of top folder (SiStrip/MechanicalView appended automatically)
    HistogramFolderName = cms.untracked.string('DQMData/'),
    #Whether to dump buffer info and raw data if any error is found: 
    #1=errors, 2=minimum info, 3=full debug with printing of the data buffer of each FED per event.
    PrintDebugMessages = cms.untracked.uint32(1)#,
    #TkmapParameters = cms.PSet(
    #    loadFedCabling = cms.untracked.bool(True),
    #    trackerdatPath = cms.untracked.string('CommonTools/TrackerMap/data/'),
    #    trackermaptxtPath = cms.untracked.string('CommonTools/TrackerMap/data/')
#        )


    )
