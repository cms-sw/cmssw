import FWCore.ParameterSet.Config as cms

# Import from the generic cfi file for CSC unpacking
from EventFilter.CSCRawToDigi.muonCSCDCCUnpacker_cfi import muonCSCDCCUnpacker

muonCSCDigis = muonCSCDCCUnpacker.clone(
    # This mask is needed by the examiner
    ExaminerMask = 0x1FEBF7F6
)

## in Run-3 include GEMs
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify( muonCSCDigis, useGEMs = False )
