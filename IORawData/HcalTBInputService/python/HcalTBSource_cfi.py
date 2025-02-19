import FWCore.ParameterSet.Config as cms

source = cms.Source("HcalTBSource",
    # Number of events to read (-1 for all events)
    maxEvents = cms.untracked.int32(100),
    #* Streams or tree branches to open.  Usual branches include
    #  HCAL_Trigger, HCAL_SlowData, HCAL_TDC, HCAL_QDC, HCAL_TDCQDC,
    #  HCAL_SourcePos, HCAL_DCC020, etc.
    #  If a stream must be remapped to a different FED id, this can be
    #  done by appending ":[new fedid]" to the branch name
    #*/
    streams = cms.untracked.vstring('HCAL_Trigger', 
        'HCAL_DCC020', 
        'HCAL_SlowData:3', 
        'HCAL_TDC:5'),
    # Files to read (can use dcap:, file:, etc)
    fileNames = cms.untracked.vstring('file:HTB_011609.root')
)


