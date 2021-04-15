import FWCore.ParameterSet.Config as cms

## baseline configuration in the class itself
from EventFilter.CSCRawToDigi.cscPackerDef_cfi import cscPackerDef
cscpacker = cscPackerDef.clone()

##
## Make changes for running in Run 2
##
# packer - simply get rid of it
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( cscpacker, useFormatVersion = cms.uint32(2013) )
run2_common.toModify( cscpacker, usePreTriggers = cms.bool(False) )
run2_common.toModify( cscpacker, packEverything = cms.bool(True) )

## in Run-3 include GEMs
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify( cscpacker, padDigiClusterTag = cms.InputTag("simMuonGEMPadDigiClusters") )
run3_GEM.toModify( cscpacker, useGEMs = cms.bool(False) )
