import FWCore.ParameterSet.Config as cms

from DPGAnalysis.SiStripTools.filtertest.MessageLogger_cff import *

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(),
                            skipBadFiles = cms.untracked.bool(True)
                            )
from DPGAnalysis.SiStripTools.filtertest.reco_69800_debug_cff import fileNames
source.fileNames = fileNames

#-------------------------------------------------------------------------
from DPGAnalysis.SiStripTools.apvlatency.fakeapvlatencyessource_cff import *
fakeapvlatency.APVLatency = cms.untracked.int32(143)
#-------------------------------------------------------------------------

from DPGAnalysis.SiStripTools.eventwithhistoryproducer_cfi import *

#------------------------------------------------------------------------
# APV Cycle Phase Producer and monitor
#------------------------------------------------------------------------
from DPGAnalysis.SiStripTools.configurableapvcyclephaseproducer_CRAFT08_cfi import *

#------------------------------------------------------------------------

sinit = cms.Sequence(consecutiveHEs + APVPhases )

from DPGAnalysis.SiStripTools.eventtimedistribution_cfi import *



