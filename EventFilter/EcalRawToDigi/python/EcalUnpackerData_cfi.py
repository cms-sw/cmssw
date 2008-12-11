import FWCore.ParameterSet.Config as cms

ecalEBunpacker = cms.EDFilter("EcalRawToDigi",
    orderedDCCIdList = cms.untracked.vint32(1, 2, 3, 4, 5, 
        6, 7, 8, 9, 10, 
        11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 
        21, 22, 23, 24, 25, 
        26, 27, 28, 29, 30, 
        31, 32, 33, 34, 35, 
        36, 37, 38, 39, 40, 
        41, 42, 43, 44, 45, 
        46, 47, 48, 49, 50, 
        51, 52, 53, 54),
    FedLabel = cms.untracked.InputTag("listfeds"),
    srpUnpacking = cms.untracked.bool(True),
    syncCheck = cms.untracked.bool(False),
    headerUnpacking = cms.untracked.bool(True),
    #untracked bool tccUnpacking    = true
    feUnpacking = cms.untracked.bool(True),
    # Default values 
    # untracked uint numbXtalTSamples    = 10
    # untracked uint numbTriggerTSamples = 1  
    # if no 2 input vectors are specified, hard coded default DCCId:FedId correspondence:
    #                                            1:601, 2:602 ...      54:654
    # is used
    # For special mappings, use the two ordered vectors here below
    #  example1:  the same as what loaded
    orderedFedList = cms.untracked.vint32(601, 602, 603, 604, 605, 
        606, 607, 608, 609, 610, 
        611, 612, 613, 614, 615, 
        616, 617, 618, 619, 620, 
        621, 622, 623, 624, 625, 
        626, 627, 628, 629, 630, 
        631, 632, 633, 634, 635, 
        636, 637, 638, 639, 640, 
        641, 642, 643, 644, 645, 
        646, 647, 648, 649, 650, 
        651, 652, 653, 654),
    #    example 2: fedId used by the DCC at th cosmics stand
    #            untracked vint32 FEDs = {13}
    #    example 3: fedId used at the best beams (e.g. SM22)
    #            untracked vint32 FEDs = {22}
    # By default these are true  
    eventPut = cms.untracked.bool(True),
    InputLabel = cms.untracked.string('source'),
    #   InputLabel = cms.untracked.string('rawDataCollector'),
    feIdCheck = cms.untracked.bool(True),
    #    by default whole ECAL fedId range is loaded, 600-670.
    #    This is in case a selection was needed.
    #    fedId==13 is special case for ECAL cosmic stand
    #    example 1: all the feds allocated to ECAL DCC's are selected by providing this empty list
    FEDs = cms.untracked.vint32(),
    #  example2:  special case cosmic stand like (SM-any in EB+01)
    # untracked vint32 orderedFedList      = {13}
    # untracked vint32 orderedDCCIdList = {28}
    #  example3:  special case test beam like (SM-22 in EB+01)
    # untracked vint32 orderedFedList      = {22}
    # untracked vint32 orderedDCCIdList = {28}
    DoRegional = cms.untracked.bool(False),
    memUnpacking = cms.untracked.bool(True),
    silentMode = cms.untracked.bool(True)
)
