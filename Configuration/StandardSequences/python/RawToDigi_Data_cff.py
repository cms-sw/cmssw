import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.RawToDigi_cff import *

RawToDigi = cms.Sequence(csctfDigis
                         +dttfDigis
                         +gctDigis
                         +gtDigis
                         +gtEvmDigis
                         +siPixelDigis
                         +siStripDigis
                         +ecalDigis
                         +ecalPreshowerDigis
                         +hcalDigis
                         +muonCSCDigis
                         +muonDTDigis
                         +muonRPCDigis
                         +castorDigis
                         +scalersRawToDigi
                         +tcdsDigis)

RawToDigi_woGCT = cms.Sequence(csctfDigis
                               +dttfDigis
                               +gtDigis
                               +gtEvmDigis
                               +siPixelDigis
                               +siStripDigis
                               +ecalDigis
                               +ecalPreshowerDigis
                               +hcalDigis
                               +muonCSCDigis
                               +muonDTDigis
                               +muonRPCDigis
                               +castorDigis
                               +scalersRawToDigi
                               +tcdsDigis)

ecalDigis.DoRegional = False

#set those back to "source"
#False by default ecalDigis.DoRegional = False


## Make changes for Run 2
##
def _modifyRawToDigiForStage1Trigger( theProcess ) :
    """
    Modifies the RawToDigi sequence if using the Stage 1 L1 trigger
    """
    theProcess.load("L1Trigger.L1TCommon.l1tRawToDigi_cfi")
    theProcess.load("L1Trigger.L1TCommon.caloStage1LegacyFormatDigis_cfi")
    # Note that this function is applied before the objects in this file are added
    # to the process. So things declared in this file should be used "bare", i.e.
    # not with "theProcess." in front of them. caloStage1Digis and caloStage1LegacyFormatDigis
    # are an exception because they are not declared in this file but loaded into the
    # process in the "load" statements above.
    L1RawToDigiSeq = cms.Sequence( gctDigis + theProcess.caloStage1Digis + theProcess.caloStage1LegacyFormatDigis)
    RawToDigi.replace( gctDigis, L1RawToDigiSeq )

# A unique name is required for this object, so I'll call it "modify<python filename>ForRun2_"
modifyConfigurationStandardSequencesRawToDigiForRun2_ = eras.stage1L1Trigger.makeProcessModifier( _modifyRawToDigiForStage1Trigger )
