import FWCore.ParameterSet.Config as cms

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"

dcsstatus = cms.EDFilter("DetStatus",
           DetectorType= cms.vstring(''),
           ApplyFilter = cms.bool(True), 
           DebugOn     = cms.untracked.bool(False), 
           AndOr       = cms.bool(True) # True=And, Flase=Or              
)

# foo bar baz
# Y7sSLa3jwJDw9
# KXfOPBE6hceuW
