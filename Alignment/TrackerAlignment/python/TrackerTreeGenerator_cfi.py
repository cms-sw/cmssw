import FWCore.ParameterSet.Config as cms

TrackerTreeGenerator = cms.EDAnalyzer('TrackerTreeGenerator',
    # default: create entry for every physical module, set to true to create one entry for the virtual double-sided module in addition
    # ask for the additional virtual module in produced TTree with "IsDoubleSide == true"
    createEntryForDoubleSidedModule = cms.bool(False)
)
