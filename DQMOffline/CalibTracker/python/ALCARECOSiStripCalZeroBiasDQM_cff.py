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

ALCARECOSiStripCalZeroBiasDQM = cms.Sequence(seqFilters*
                                             SiStripCalZeroBiasMonitorCluster)
