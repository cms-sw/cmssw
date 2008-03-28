import FWCore.ParameterSet.Config as cms

# This is a generic cfi file for CSC Digi to Raw packing
# tumanov@rice.edu 3/16/07
cscpacker = cms.EDFilter("CSCDigiToRawModule",
    # this is a label for digi input (to be replaced by InputTag in 140 and up)	
    DigiCreator = cms.untracked.string('muonCSCDigis')
)


