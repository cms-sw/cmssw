#
# cfi file for a module to produce raw GT DAQ starting from a text (ASCII) file
#

import FWCore.ParameterSet.Config as cms

l1GtTextToRaw = cms.EDProducer("L1GtTextToRaw",

    # type of the text file
    TextFileType = cms.untracked.string('VmeSpyDump'),
    
    # name of the text file to be packed
    # the module is using a EmptySource source
    TextFileName = cms.untracked.string('testGt_TextToRaw_source.txt'),

    # FED Id for GT DAQ record 
    # default value defined in DataFormats/FEDRawData/src/FEDNumbering.cc
    DaqGtFedId = cms.untracked.int32(813),
    
    # FED raw data size (in 8bits units, including header and trailer)
    # If negative value, the size is retrieved from the trailer.    
    RawDataSize = cms.untracked.int32(872)
)


