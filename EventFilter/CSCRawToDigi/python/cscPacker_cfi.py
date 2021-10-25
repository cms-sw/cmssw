import FWCore.ParameterSet.Config as cms

## baseline configuration in the class itself
from EventFilter.CSCRawToDigi.cscPackerDef_cfi import cscPackerDef
cscpacker = cscPackerDef.clone()

## In Run-2 common: update the format version for new OTMBs in ME1/1
## Note: in the past, the packing with triggers and pretriggers was disabled
## for Run-2, Run-3 and Phase-2 scenarios. This should no longer be the case
## as of CMSSW_12_0_0_pre5
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( cscpacker,
                      formatVersion = 2013)

## in Run-3 scenarios with GEM: pack GEM clusters
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify( cscpacker,
                   useGEMs = True)
