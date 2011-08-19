def customiseMCDBCheck(process):

    process.MessageLogger.categories=cms.untracked.vstring('FwkJob'
                                                           ,'FwkReport'
                                                           ,'FwkSummary'
                                                           ,'Root_NoDictionary'
                                                           ,'Generator'
                                                           ,'LHEInterface'
                                                           )
    

    process.MessageLogger.cerr.INFO = cms.untracked.PSet(limit = cms.untracked.int32(-1))
    process.MessageLogger.cerr.Generator = cms.untracked.PSet(limit = cms.untracked.int32(0))
    process.MessageLogger.cerr.LHEInterface = cms.untracked.PSet(limit = cms.untracked.int32(10000))
    process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10000)

    #Add these 3 lines to put back the summary for timing information at the end of the logfile
    #(needed for TimeReport report)
    process.options = cms.untracked.PSet(
        wantSummary = cms.untracked.bool(True)
        )

        
    return(process)
