import FWCore.ParameterSet.Config as cms

siStripBuildTrackerMap = cms.EDAnalyzer(
    "BuildTrackerMapPlugin",
    #input root file containing histograms
    InputFileName = cms.string('DQMStore.root'),
    DoDifference = cms.bool(False),
    InputFileNameForDiff = cms.string('DQMStore.root'),
    #name of tkHistoMap to dump
    TkHistoMapNameVec = cms.vstring('TkHMap_MeanCMAPV0','TkHMap_MeanCMAPV1','TkHMap_MeanCMAPV0minusAPV1','TkHMap_RmsCMAPV0','TkHMap_RmsCMAPV1','TkHMap_RmsCMAPV0minusAPV1'),
    MinValueVec = cms.vdouble(120,120,-20,0,0,0),
    MaxValueVec = cms.vdouble(140,140,20,10,10,10),
    MechanicalView = cms.bool(True),
    #Name of top folder (SiStrip/MechanicalView appended automatically)
    HistogramFolderName = cms.string('DQMData/'),
    #Whether to dump buffer info and raw data if any error is found: 
    #1=errors, 2=minimum info, 3=full debug with printing of the data buffer of each FED per event.
    PrintDebugMessages = cms.uint32(1),
    TkmapParameters = cms.PSet(
        loadFedCabling = cms.bool(True),
    #    trackerdatPath = cms.string('CommonTools/TrackerMap/data/'),
    #    trackermaptxtPath = cms.string('CommonTools/TrackerMap/data/')
        )
    )
