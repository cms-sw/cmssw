import FWCore.ParameterSet.Config as cms


# This function makes all the changes to cscTriggerPrimitiveDigis required
# for it to work in Run 2. It is declared here to be obvious when this file
# is opened, but not applied until after cscTriggerPrimitiveDigis is
# declared (and then only if the "run2" era is active).
def _modifyCscTriggerPrimitiveDigisForRun2( object ) :
    """
    Modifies cscTriggerPrimitiveDigis for Run 2
    """
    object.debugParameters = True
    object.checkBadChambers = False
    object.commonParam.isSLHC = True
    object.commonParam.smartME1aME1b = True
    object.commonParam.gangedME1a = False
    object.alctParam07.alctNarrowMaskForR1 = True
    object.alctParam07.alctGhostCancellationBxDepth = cms.int32(1)
    object.alctParam07.alctGhostCancellationSideQuality = cms.bool(True)
    object.alctParam07.alctPretrigDeadtime = cms.uint32(4)
    object.clctParam07.clctPidThreshPretrig = 4
    object.clctParam07.clctMinSeparation = 5
    object.tmbParam.matchTrigWindowSize = 3


def _modifyCscTriggerPrimitiveDigisForRun2GE11( object ) :
    """
    Modifies cscTriggerPrimitiveDigis for Run 2 + GEMs
    """
    object.GEMPadDigiProducer = "simMuonGEMPadDigis"
    object.commonParam.runME11ILT = cms.bool(False)
    object.clctSLHC.clctNplanesHitPattern = 3
    object.clctSLHC.clctPidThreshPretrig = 2
    
    # to be used by ME11 chambers with GEM-CSC ILT
    me11tmbSLHCGEM = cms.PSet(
        mpcBlockMe1a    = cms.uint32(0),
        alctTrigEnable  = cms.uint32(0),
        clctTrigEnable  = cms.uint32(0),
        matchTrigEnable = cms.uint32(1),
        matchTrigWindowSize = cms.uint32(3),
        tmbL1aWindowSize = cms.uint32(7),
        verbosity = cms.int32(0),
        tmbEarlyTbins = cms.int32(4),
        tmbReadoutEarliest2 = cms.bool(False),
        tmbDropUsedAlcts = cms.bool(False),
        clctToAlct = cms.bool(False),
        tmbDropUsedClcts = cms.bool(False),
        matchEarliestAlctME11Only = cms.bool(False),
        matchEarliestClctME11Only = cms.bool(False),
        tmbCrossBxAlgorithm = cms.uint32(2),
        maxME11LCTs = cms.uint32(2),

        ## run in debug mode
        debugLUTs = cms.bool(False),
        debugMatching = cms.bool(False),
        debugGEMDphi = cms.bool(False),

        ## use old dataformat
        useOldLCTDataFormat = cms.bool(True),

        ## copad construction
        maxDeltaBXInCoPad = cms.int32(1),
        maxDeltaPadInCoPad = cms.int32(1),

        ## matching to pads in case LowQ CLCT
        maxDeltaBXPadEven = cms.int32(1),
        maxDeltaBXPadOdd = cms.int32(1),
        maxDeltaPadPadEven = cms.int32(2),
        maxDeltaPadPadOdd = cms.int32(3),

        ## matching to pads in case absent CLCT
        maxDeltaBXCoPadEven = cms.int32(0),
        maxDeltaBXCoPadOdd = cms.int32(0),
        maxDeltaPadCoPadEven = cms.int32(2),
        maxDeltaPadCoPadOdd = cms.int32(3),

        ## efficiency recovery switches
        dropLowQualityCLCTsNoGEMs_ME1a = cms.bool(False),
        dropLowQualityCLCTsNoGEMs_ME1b = cms.bool(True),
        dropLowQualityALCTsNoGEMs_ME1a = cms.bool(False),
        dropLowQualityALCTsNoGEMs_ME1b = cms.bool(False),
        buildLCTfromALCTandGEM_ME1a = cms.bool(True),
        buildLCTfromALCTandGEM_ME1b = cms.bool(True),
        buildLCTfromCLCTandGEM_ME1a = cms.bool(False),
        buildLCTfromCLCTandGEM_ME1b = cms.bool(False),
        doLCTGhostBustingWithGEMs = cms.bool(False),
        correctLCTtimingWithGEM = cms.bool(False),
        promoteALCTGEMpattern = cms.bool(True),
        promoteALCTGEMquality = cms.bool(True),
        promoteCLCTGEMquality_ME1a = cms.bool(True),
        promoteCLCTGEMquality_ME1b = cms.bool(True),
        
        ## rate reduction 
        doGemMatching = cms.bool(True),
        gemMatchDeltaEta = cms.double(0.08),
        gemMatchDeltaBX = cms.int32(1),
        gemMatchDeltaPhiOdd = cms.double(1),
        gemMatchDeltaPhiEven = cms.double(1),
        gemMatchMinEta = cms.double(1.55),
        gemMatchMaxEta = cms.double(2.15),
        gemClearNomatchLCTs = cms.bool(False),

        ## cross BX algorithm
        firstTwoLCTsInChamber = cms.bool(True),
    )

    object.me11tmbSLHCGEM = me11tmbSLHCGEM

def _modifyCscTriggerPrimitiveDigisForRun4GE21( object ) :
    """
    Modifies cscTriggerPrimitiveDigis for Run 4 + GEMs
    """
    object.commonParam.runME21ILT = cms.bool(False)
    object.clctSLHCME21 = object.clctSLHC.clone()
    object.alctSLHCME21 = object.alctSLHC.clone()
    object.alctSLHCME21.alctNplanesHitPattern = 3
    object.alctSLHCME21.clctNplanesHitPattern = 3
    object.alctSLHCME21.clctPidThreshPretrig = 2

    # to be used by ME21 chambers with GEM-CSC ILT
    me21tmbSLHCGEM = cms.PSet(
        mpcBlockMe1a    = cms.uint32(0),
        alctTrigEnable  = cms.uint32(0),
        clctTrigEnable  = cms.uint32(0),
        matchTrigEnable = cms.uint32(1),
        matchTrigWindowSize = cms.uint32(3),
        tmbL1aWindowSize = cms.uint32(7),
        verbosity = cms.int32(0),
        tmbEarlyTbins = cms.int32(4),
        tmbReadoutEarliest2 = cms.bool(False),
        tmbDropUsedAlcts = cms.bool(False),
        clctToAlct = cms.bool(False),
        tmbDropUsedClcts = cms.bool(False),
        matchEarliestAlctME21Only = cms.bool(False),
        matchEarliestClctME21Only = cms.bool(False),
        tmbCrossBxAlgorithm = cms.uint32(2),
        maxME21LCTs = cms.uint32(2),

        ## run in debug mode
        debugLUTs = cms.bool(False),
        debugMatching = cms.bool(False),
        debugGEMDphi = cms.bool(False),

        ## use old dataformat
        useOldLCTDataFormat = cms.bool(True),

        ## copad construction
        maxDeltaBXInCoPad = cms.int32(1),
        maxDeltaPadInCoPad = cms.int32(2),

        ## matching to pads in case LowQ CLCT
        maxDeltaBXPad = cms.int32(1),
        maxDeltaPadPadOdd = cms.int32(4),
        maxDeltaPadPadEven = cms.int32(3),
        maxDeltaWg = cms.int32(2),

        ## matching to pads in case absent CLCT
        maxDeltaBXCoPad = cms.int32(1),
        maxDeltaPadCoPad = cms.int32(2),

        ## efficiency recovery switches
        dropLowQualityALCTsNoGEMs = cms.bool(False),
        dropLowQualityCLCTsNoGEMs = cms.bool(True),
        buildLCTfromALCTandGEM = cms.bool(True),
        buildLCTfromCLCTandGEM = cms.bool(False),
        doLCTGhostBustingWithGEMs = cms.bool(False),
        correctLCTtimingWithGEM = cms.bool(False),
        promoteALCTGEMpattern = cms.bool(True),
        promoteALCTGEMquality = cms.bool(True),
        promoteCLCTGEMquality = cms.bool(True),

        ## rate reduction 
        doGemMatching = cms.bool(True),
        gemMatchDeltaEta = cms.double(0.08),
        gemMatchDeltaBX = cms.int32(1),
        gemMatchDeltaPhiOdd = cms.double(1),
        gemMatchDeltaPhiEven = cms.double(1),
        gemMatchMinEta = cms.double(1.5),
        gemMatchMaxEta = cms.double(2.45),
        gemClearNomatchLCTs = cms.bool(False),

        firstTwoLCTsInChamber = cms.bool(True),
    )

    object.me21tmbSLHCGEM = me21tmbSLHCGEM

def _modifyCscTriggerPrimitiveDigisForRun4RPC( object ) :
    """
    Modifies cscTriggerPrimitiveDigis for Run 4 + RPCs
    """
    object.RPCDigiProducer = "simMuonRPCDigis"
    object.commonParam.runME3141ILT = cms.bool(False)
    object.clctSLHCME3141 = object.clctSLHC.clone()
    object.alctSLHCME3141 = object.alctSLHC.clone()
    object.alctSLHCME3141.alctNplanesHitPattern = 3
    object.alctSLHCME3141.clctNplanesHitPattern = 3
    object.alctSLHCME3141.clctPidThreshPretrig = 2

    # to be used by ME31-ME41 chambers with RPC-CSC ILT
    me3141tmbSLHCRPC = cms.PSet(
        mpcBlockMe1a    = cms.uint32(0),
        alctTrigEnable  = cms.uint32(0),
        clctTrigEnable  = cms.uint32(0),
        matchTrigEnable = cms.uint32(1),
        matchTrigWindowSize = cms.uint32(3),
        tmbL1aWindowSize = cms.uint32(7),
        verbosity = cms.int32(0),
        tmbEarlyTbins = cms.int32(4),
        tmbReadoutEarliest2 = cms.bool(False),
        tmbDropUsedAlcts = cms.bool(False),
        clctToAlct = cms.bool(False),
        tmbDropUsedClcts = cms.bool(False),
        matchEarliestClctME3141Only = cms.bool(False),
        tmbCrossBxAlgorithm = cms.uint32(2),
        maxME3141LCTs = cms.uint32(2),

        ## run in debug mode
        debugLUTs = cms.bool(False),
        debugMatching = cms.bool(False),

        ## use old dataformat
        useOldLCTDataFormatC = cms.bool(True),

        ## matching to digis in case LowQ CLCT
        maxDeltaBXRPC = cms.int32(0),
        maxDeltaStripRPCOdd = cms.int32(6),
        maxDeltaStripRPCEven = cms.int32(4),
        maxDeltaWg = cms.int32(2),

        ## efficiency recovery switches
        dropLowQualityCLCTsNoRPCs = cms.bool(True),
        buildLCTfromALCTandRPC = cms.bool(True),
        buildLCTfromCLCTandRPC = cms.bool(False),
        buildLCTfromLowQstubandRPC = cms.bool(True),
        promoteCLCTRPCquality = cms.bool(True),
        promoteALCTRPCpattern = cms.bool(True),
        promoteALCTRPCquality = cms.bool(True),

        ## rate reduction 
        doRpcMatching = cms.bool(True),
        rpcMatchDeltaEta = cms.double(0.08),
        rpcMatchDeltaBX = cms.int32(1),
        rpcMatchDeltaPhiOdd = cms.double(1),
        rpcMatchDeltaPhiEven = cms.double(1),
        rpcMatchMinEta = cms.double(1.5),
        rpcClearNomatchLCTs = cms.bool(False),
    ),

    object.me3141tmbSLHCRPC = me3141tmbSLHCRPC

from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
# Default parameters for CSCTriggerPrimitives generator
# =====================================================
cscTriggerPrimitiveDigis = cms.EDProducer("CSCTriggerPrimitivesProducer",
    CSCCommonTrigger,

    # if False, parameters will be read in from DB using EventSetup mechanism
    # else will use parameters from this config
    debugParameters = cms.bool(False),
    
    # Name of digi producer module(s)
    CSCComparatorDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    CSCWireDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    GEMPadDigiProducer = cms.InputTag(""),
    RPCDigiProducer = cms.InputTag(""),

    # for SLHC studies we don't want bad chambers checks so far
    checkBadChambers = cms.bool(True),

    # Parameters common for all boards
    commonParam = cms.PSet(
        isTMB07 = cms.bool(True),
        isMTCC = cms.bool(False),
        
        # Flag for SLHC studies (upgraded ME11, MPC)
        # (if true, isTMB07 should be true as well)
        isSLHC = cms.bool(False),

        # ME1a configuration:
        # smartME1aME1b=f, gangedME1a=t
        #   default logic for current HW
        # smartME1aME1b=t, gangedME1a=f
        #   realistic upgrade scenario: 
        #   one ALCT finder and two CLCT finders per ME11
        #   with additional logic for A/CLCT matching with ME1a unganged
        # smartME1aME1b=t, gangedME1a=t
        #   the previous case with ME1a still being ganged
        # Note: gangedME1a has effect only if smartME1aME1b=t
        smartME1aME1b = cms.bool(False),
        gangedME1a = cms.bool(True),
        
        # flags to optionally disable finding stubs in ME42 or ME1a
        disableME1a = cms.bool(False),
        disableME42 = cms.bool(False),

        ## enable the GEM-CSC integrated triggers for ME11 or ME21
        runME11ILT = cms.bool(False),
        runME21ILT = cms.bool(False),
        runME3141ILT = cms.bool(False),
    ),

    # Parameters for ALCT processors: old MC studies
    alctParamOldMC = cms.PSet(
        alctFifoTbins   = cms.uint32(16),
        alctFifoPretrig = cms.uint32(10),
        alctDriftDelay  = cms.uint32(3),
        alctNplanesHitPretrig = cms.uint32(2),
        alctNplanesHitPattern = cms.uint32(4),
        alctNplanesHitAccelPretrig = cms.uint32(2),
        alctNplanesHitAccelPattern = cms.uint32(4),
        alctTrigMode       = cms.uint32(3),
        alctAccelMode      = cms.uint32(1),
        alctL1aWindowWidth = cms.uint32(5),
        verbosity = cms.int32(0)
    ),

    # Parameters for ALCT processors: MTCC-II
    alctParamMTCC = cms.PSet(
        alctFifoTbins   = cms.uint32(16),
        alctFifoPretrig = cms.uint32(10),
        alctDriftDelay  = cms.uint32(3),
        alctNplanesHitPretrig = cms.uint32(2),
        alctNplanesHitPattern = cms.uint32(4),
        alctNplanesHitAccelPretrig = cms.uint32(2),
        alctNplanesHitAccelPattern = cms.uint32(4),
        alctTrigMode       = cms.uint32(2),
        alctAccelMode      = cms.uint32(0),
        alctL1aWindowWidth = cms.uint32(3),
        verbosity = cms.int32(0)
    ),

    # Parameters for ALCT processors: 2007 and later
    alctParam07 = cms.PSet(
        alctFifoTbins   = cms.uint32(16),
        alctFifoPretrig = cms.uint32(10),
        alctDriftDelay  = cms.uint32(2),
        alctNplanesHitPretrig = cms.uint32(3),
        alctNplanesHitPattern = cms.uint32(4),
        alctNplanesHitAccelPretrig = cms.uint32(3),
        alctNplanesHitAccelPattern = cms.uint32(4),
        alctTrigMode       = cms.uint32(2),
        alctAccelMode      = cms.uint32(0),
        alctL1aWindowWidth = cms.uint32(7),
        verbosity = cms.int32(0),

        # Configure early_tbins instead of hardcoding it
        alctEarlyTbins = cms.int32(4),

        # Use narrow pattern mask for ring 1 chambers
        alctNarrowMaskForR1 = cms.bool(False),

        # configured, not hardcoded, hit persistency
        alctHitPersist  = cms.uint32(6),

        # configure, not hardcode, up to how many BXs in the past
        # ghost cancellation in neighboring WGs may happen
        alctGhostCancellationBxDepth = cms.int32(4),

        # whether to compare the quality of stubs in neighboring WGs in the past
        # to the quality of a stub in current WG 
        # when doing ghost cancellation 
        alctGhostCancellationSideQuality = cms.bool(False),
        
        # how soon after pretrigger and alctDriftDelay can next pretrigger happen?
        alctPretrigDeadtime = cms.uint32(4),
        
        # SLHC only for ME11:
        # whether to store the "corrected" ALCT stub time 
        # (currently it is median time of particular hits in a pattern) into the ASCCLCTDigi bx,
        # and temporary store the regular "key layer hit" time into the CSCCLCTDigi fullBX:
        alctUseCorrectedBx = cms.bool(False)
    ),

    # Parameters for ALCT processors: SLHC studies
    alctSLHC = cms.PSet(
        alctFifoTbins   = cms.uint32(16),
        alctFifoPretrig = cms.uint32(10),
        alctDriftDelay  = cms.uint32(2),
        alctNplanesHitPretrig = cms.uint32(3),
        alctNplanesHitPattern = cms.uint32(4),
        alctNplanesHitAccelPretrig = cms.uint32(3),
        alctNplanesHitAccelPattern = cms.uint32(4),
        alctTrigMode       = cms.uint32(2),
        alctAccelMode      = cms.uint32(0),
        alctL1aWindowWidth = cms.uint32(7),
        verbosity = cms.int32(0),

        # Configure early_tbins instead of hardcoding it
        alctEarlyTbins = cms.int32(4),

        # Use narrow pattern mask for ring 1 chambers
        alctNarrowMaskForR1 = cms.bool(True),

        # configured, not hardcoded, hit persistency
        alctHitPersist  = cms.uint32(6),

        # configure, not hardcode, up to how many BXs in the past
        # ghost cancellation in neighboring WGs may happen
        alctGhostCancellationBxDepth = cms.int32(1),
        
        # whether to compare the quality of stubs in neighboring WGs in the past
        # to the quality of a stub in current WG 
        # when doing ghost cancellation 
        alctGhostCancellationSideQuality = cms.bool(True),
        
        # how soon after pretrigger and alctDriftDelay can next pretrigger happen?
        alctPretrigDeadtime = cms.uint32(0),
        
        # SLHC only for ME11:
        # whether to store the "corrected" ALCT stub time 
        # (currently it is median time of particular hits in a pattern) into the ASCCLCTDigi bx,
        # and temporary store the regular "key layer hit" time into the CSCCLCTDigi fullBX:
        alctUseCorrectedBx = cms.bool(True)
    ),

    # Parameters for CLCT processors: old MC studies
    clctParamOldMC = cms.PSet(
        clctFifoTbins   = cms.uint32(12),
        clctFifoPretrig = cms.uint32(7),
        clctHitPersist  = cms.uint32(6),
        clctDriftDelay  = cms.uint32(2),
        clctNplanesHitPretrig = cms.uint32(2),
        clctNplanesHitPattern = cms.uint32(4),
        clctPidThreshPretrig  = cms.uint32(2),
        clctMinSeparation     = cms.uint32(10),
        # Debug
        verbosity = cms.int32(0)
    ),

    # Parameters for CLCT processors: MTCC-II
    clctParamMTCC = cms.PSet(
        clctFifoTbins   = cms.uint32(12),
        clctFifoPretrig = cms.uint32(7),
        clctHitPersist  = cms.uint32(6),
        clctDriftDelay  = cms.uint32(2),
        clctNplanesHitPretrig = cms.uint32(4),
        clctNplanesHitPattern = cms.uint32(1),
        clctPidThreshPretrig  = cms.uint32(2),
        clctMinSeparation     = cms.uint32(10),
        # Debug
        verbosity = cms.int32(0)
    ),

    # Parameters for CLCT processors: 2007 and later
    clctParam07 = cms.PSet(
        clctFifoTbins   = cms.uint32(12),
        clctFifoPretrig = cms.uint32(7),
        clctHitPersist  = cms.uint32(4),
        clctDriftDelay  = cms.uint32(2),
        clctNplanesHitPretrig = cms.uint32(3),
        clctNplanesHitPattern = cms.uint32(4),
        clctPidThreshPretrig  = cms.uint32(2),
        clctMinSeparation     = cms.uint32(10),
        # Debug
        verbosity = cms.int32(0),

        # BX to start CLCT finding (poor man's dead-time shortening):
        clctStartBxShift  = cms.int32(0)
    ),

    # Parameters for CLCT processors: SLHC studies
    clctSLHC = cms.PSet(
        clctFifoTbins   = cms.uint32(12),
        clctFifoPretrig = cms.uint32(7),
        clctHitPersist  = cms.uint32(4),
        clctDriftDelay  = cms.uint32(2),
        clctNplanesHitPretrig = cms.uint32(3),
        clctNplanesHitPattern = cms.uint32(4),
        # increase pattern ID threshold from 2 to 4 to trigger higher pt tracks  
        clctPidThreshPretrig  = cms.uint32(4),
        # decrease possible minimal #HS distance between two CLCTs in a BX from 10 to 5:
        clctMinSeparation     = cms.uint32(5),
        # Debug
        verbosity = cms.int32(0),

        # BX to start CLCT finding (poor man's to shorten the dead-time):
        clctStartBxShift  = cms.int32(0),
        
        # Turns on algorithms of localized dead-time zones:
        useDeadTimeZoning = cms.bool(True),
        
        # Width (in #HS) of a fixed dead zone around a key HS:
        clctStateMachineZone = cms.uint32(8),
        
        # Enables the algo which instead of using the fixed dead zone width, 
        # varies it depending on the width of a triggered CLCT pattern
        # (if True, the clctStateMachineZone is ignored):
        useDynamicStateMachineZone = cms.bool(True),
        
        # Pretrigger HS +- clctPretriggerTriggerZone sets the trigger matching zone
        # which defines how far from pretrigger HS the TMB may look for a trigger HS
        # (it becomes important to do so with localized dead-time zoning): 
        clctPretriggerTriggerZone = cms.uint32(5),
        
        # whether to store the "corrected" CLCT stub time 
        # (currently it is median time of all hits in a pattern) into the CSCCLCTDigi bx,
        # and temporary store the regular "key layer hit" time into the CSCCLCTDigi fullBX:
        clctUseCorrectedBx = cms.bool(True)
    ),

    tmbParam = cms.PSet(
        mpcBlockMe1a    = cms.uint32(0),
        alctTrigEnable  = cms.uint32(0),
        clctTrigEnable  = cms.uint32(0),
        matchTrigEnable = cms.uint32(1),
        matchTrigWindowSize = cms.uint32(7),
        tmbL1aWindowSize = cms.uint32(7),
        # Debug
        verbosity = cms.int32(0),

        # Configure early_tbins instead of hardcoding it
        tmbEarlyTbins = cms.int32(4),
        
        # Flag for whether to readout only the earliest max two LCTs in a 
        # L1A readout window, as there is only room just for two in the TMB header.
        # If false, all LCTs would be readout in L1A window.
        tmbReadoutEarliest2 = cms.bool(True),

        # For CLCT-centric matching, whether to drop ALCTs that were matched
        # to CLCTs in this BX, and not use them in the following BX 
        # (default non-upgrade TMB behavior).
        tmbDropUsedAlcts = cms.bool(True)
    ),

    # to be used by ME11 chambers with upgraded TMB and ALCT
    tmbSLHC = cms.PSet(
        mpcBlockMe1a    = cms.uint32(0),
        alctTrigEnable  = cms.uint32(0),
        clctTrigEnable  = cms.uint32(0),
        matchTrigEnable = cms.uint32(1),
        # reduce ALCT-CLCT matching window size from 7 to 3
        matchTrigWindowSize = cms.uint32(3),
        tmbL1aWindowSize = cms.uint32(7),
        # Debug
        verbosity = cms.int32(0),

        # Configure early_tbins instead of hardcoding it
        tmbEarlyTbins = cms.int32(4),
        
        # Flag for whether to readout only the earliest max two LCTs in a 
        # L1A readout window, as there is only room just for two in the TMB header.
        # If false, all LCTs would be readout in L1A window.
        tmbReadoutEarliest2 = cms.bool(False),

        # For CLCT-centric matching, whether to drop ALCTs that were matched
        # to CLCTs in this BX, and not use them in the following BX 
        # (default non-upgrade TMB behavior).
        tmbDropUsedAlcts = cms.bool(False),

        # Switch to enable
        #  True = CLCT-centric matching (default non-upgrade behavior, 
        #         take CLCTs in BX look for matching ALCTs in window)
        #  False = ALCT-centric matching (recommended for SLHC, 
        #         take ALCTs in BX look for matching CLCTs in window)
        clctToAlct = cms.bool(False),

        # For ALCT-centric matching, whether to drop CLCTs that were matched
        # to ALCTs in this BX, and not use them in the following BX 
        tmbDropUsedClcts = cms.bool(False),
        
        # For CLCT-centric matching in ME11, break after finding 
        # the first BX with matching ALCT 
        matchEarliestAlctME11Only = cms.bool(False),
        
        # For ALCT-centric matching in ME11, break after finding 
        # the first BX with matching CLCT 
        matchEarliestClctME11Only = cms.bool(False),
        
        # 0 = default "non-X-BX" sorting algorithm, 
        #     where the first BX with match goes first
        # 1 = simple X-BX sorting algorithm, 
        #     where the central match BX goes first, 
        #     then the closest early, the slocest late, etc.
        tmbCrossBxAlgorithm = cms.uint32(1),
        
        # How many maximum LCTs per whole chamber per BX to keep
        # (supposedly, 1b and 1a can have max 2 each)
        maxME11LCTs = cms.uint32(2)
    ),

    # MPC sorter config for SLHC studies
    mpcSLHC = cms.PSet(
        mpcMaxStubs = cms.uint32(3)
    )
)

#
# If the run2 era is active, make the required changes
#
from Configuration.StandardSequences.Eras import eras
eras.run2_common.toModify( cscTriggerPrimitiveDigis, _modifyCscTriggerPrimitiveDigisForRun2 )
