import FWCore.ParameterSet.Config as cms

from ..modules.bmtfDigis_cfi import *
from ..modules.caloLayer1Digis_cfi import *
from ..modules.caloStage2Digis_cfi import *
from ..modules.emtfStage2Digis_cfi import *
from ..modules.gmtStage2Digis_cfi import *
from ..modules.gtStage2Digis_cfi import *
from ..modules.omtfStage2Digis_cfi import *
from ..modules.rpcCPPFRawToDigi_cfi import *
from ..modules.rpcTwinMuxRawToDigi_cfi import *
from ..modules.twinMuxStage2Digis_cfi import *

L1TRawToDigi_Stage2 = cms.Task(bmtfDigis, caloLayer1Digis, caloStage2Digis, emtfStage2Digis, gmtStage2Digis, gtStage2Digis, omtfStage2Digis, rpcCPPFRawToDigi, rpcTwinMuxRawToDigi, twinMuxStage2Digis)
