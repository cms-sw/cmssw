import FWCore.ParameterSet.Config as cms

## baseline configuration in the class itself
from EventFilter.CSCRawToDigi.cscPackerDef_cfi import cscPackerDef
cscpacker = cscPackerDef.clone()

##
## Make changes for running in Run 2
##
# packer - simply get rid of it
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( cscpacker,
                      useFormatVersion = 2013,
                      usePreTriggers = False,
                      packEverything = True)

## in Run-3 include GEMs
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify( cscpacker,
                   padDigiClusterTag = "simMuonGEMPadDigiClusters",
                   useGEMs = False)
