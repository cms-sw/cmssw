import FWCore.ParameterSet.Config as cms

# cfi for L1 GT VHDL Writer
l1GtVhdlWriter = cms.EDAnalyzer("L1GtVhdlWriter",
                              
    # choose VHDL directory
    VhdlTemplatesDir = cms.string('../data/VhdlTemplates/'),
    
    # choose ouput directory
    OutputDir = cms.string('../test/output/')
)


