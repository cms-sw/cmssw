import FWCore.ParameterSet.Config as cms

#------------------------
# SiStripMonitorCluster #
#------------------------

from DQM.SiStripMonitorCluster.SiStripMonitorClusterAlca_cfi import *
SiStripCalZeroBiasMonitorCluster.TopFolderName = cms.string('AlCaReco/SiStrip')

#--------------------------------------------------------------------------
# Filters to remove APV induced noisy events and high multiplicity events #
#--------------------------------------------------------------------------

from DPGAnalysis.SiStripTools.FilterSequenceForAlCaRecoDQM_cfi import *

#------------
# Sequence #
#------------

# Short-term workaround to preserve the "run for every event" while removing the use of convertToUnscheduled()
# To be reverted in a subsequent PR
ALCARECOSiStripCalZeroBiasDQMTask = cms.Task(SiStripCalZeroBiasMonitorCluster)
ALCARECOSiStripCalZeroBiasDQM = cms.Sequence(seqFilters,
                                             ALCARECOSiStripCalZeroBiasDQMTask)
