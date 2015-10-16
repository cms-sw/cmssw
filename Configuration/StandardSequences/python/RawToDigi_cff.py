import FWCore.ParameterSet.Config as cms

# This object is used to selectively make changes for different running
# scenarios. In this case it makes changes for Run 2.
from Configuration.StandardSequences.Eras import eras

from CondCore.DBCommon.CondDBSetup_cfi import *

import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
csctfDigis = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone()

import EventFilter.DTTFRawToDigi.dttfunpacker_cfi
dttfDigis = EventFilter.DTTFRawToDigi.dttfunpacker_cfi.dttfunpacker.clone()

import EventFilter.GctRawToDigi.l1GctHwDigis_cfi
gctDigis = EventFilter.GctRawToDigi.l1GctHwDigis_cfi.l1GctHwDigis.clone()

import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()

import EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi
gtEvmDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi.l1GtEvmUnpack.clone()

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *

from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *

from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *

import EventFilter.EcalRawToDigi.EcalUnpackerData_cfi
ecalDigis = EventFilter.EcalRawToDigi.EcalUnpackerData_cfi.ecalEBunpacker.clone()

import EventFilter.ESRawToDigi.esRawToDigi_cfi
ecalPreshowerDigis = EventFilter.ESRawToDigi.esRawToDigi_cfi.esRawToDigi.clone()

import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
hcalDigis = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()

import EventFilter.CSCRawToDigi.cscUnpacker_cfi
muonCSCDigis = EventFilter.CSCRawToDigi.cscUnpacker_cfi.muonCSCDigis.clone()

import EventFilter.DTRawToDigi.dtunpacker_cfi
muonDTDigis = EventFilter.DTRawToDigi.dtunpacker_cfi.muonDTDigis.clone()

import EventFilter.RPCRawToDigi.rpcUnpacker_cfi
muonRPCDigis = EventFilter.RPCRawToDigi.rpcUnpacker_cfi.rpcunpacker.clone()

from EventFilter.CastorRawToDigi.CastorRawToDigi_cff import *
castorDigis = EventFilter.CastorRawToDigi.CastorRawToDigi_cfi.castorDigis.clone( FEDs = cms.untracked.vint32(690,691,692) )

from EventFilter.ScalersRawToDigi.ScalersRawToDigi_cfi import *

from EventFilter.Utilities.tcdsRawToDigi_cfi import *
tcdsDigis = EventFilter.Utilities.tcdsRawToDigi_cfi.tcdsRawToDigi.clone()

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

RawToDigi_noTk = cms.Sequence(csctfDigis
                              +dttfDigis
                              +gctDigis
                              +gtDigis
                              +gtEvmDigis
                              +ecalDigis
                              +ecalPreshowerDigis
                              +hcalDigis
                              +muonCSCDigis
                              +muonDTDigis
                              +muonRPCDigis
                              +castorDigis
                              +scalersRawToDigi
                              +tcdsDigis)
    
scalersRawToDigi.scalersInputTag = 'rawDataCollector'
csctfDigis.producer = 'rawDataCollector'
dttfDigis.DTTF_FED_Source = 'rawDataCollector'
gctDigis.inputLabel = 'rawDataCollector'
gtDigis.DaqGtInputTag = 'rawDataCollector'
siPixelDigis.InputLabel = 'rawDataCollector'
#false by default anyways ecalDigis.DoRegional = False
ecalDigis.InputLabel = 'rawDataCollector'
ecalPreshowerDigis.sourceTag = 'rawDataCollector'
hcalDigis.InputLabel = 'rawDataCollector'
muonCSCDigis.InputObjects = 'rawDataCollector'
muonDTDigis.inputLabel = 'rawDataCollector'
muonRPCDigis.InputLabel = 'rawDataCollector'
gtEvmDigis.EvmGtInputTag = 'rawDataCollector'
castorDigis.InputLabel = 'rawDataCollector'

##
## Make changes for Run 2
##
def _modifyRawToDigiForRun2( RawToDigi_object ) :
    RawToDigi_object.remove(gtEvmDigis)

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

eras.run2_common.toModify( RawToDigi, func=_modifyRawToDigiForRun2 )
# A unique name is required for this object, so I'll call it "modify<python filename>ForRun2_"
modifyConfigurationStandardSequencesRawToDigiForRun2_ = eras.stage1L1Trigger.makeProcessModifier( _modifyRawToDigiForStage1Trigger )
