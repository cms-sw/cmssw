import FWCore.ParameterSet.Config as cms
from DQMServices.Core.nanoDQMIO_perLSoutput_cff import *

DQMStore = cms.Service("DQMStore",
    verbose = cms.untracked.int32(0),
    # similar to LSBasedMode but for offline. Explicitly sets LumiFLag on all
    # MEs/modules that allow it (canSaveByLumi)
    saveByLumi = cms.untracked.bool(False),
    #Following list has no effect if saveByLumi is False
    MEsToSave = cms.untracked.vstring(nanoDQMIO_perLSoutput.MEsToSave),  
    trackME = cms.untracked.string(""),
    #Legacy code should be out. Suggested on:
    #https://github.com/cms-sw/cmssw/pull/34231#issuecomment-874789622
    assertLegacySafe  = cms.untracked.bool(False),
    # UNUSED: historical HLT configs expect this option to be present, so it
    # remains here, even though the DQMStore does not use it any more.
    enableMultiThread = cms.untracked.bool(True)
)
