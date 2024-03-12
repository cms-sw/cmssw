import FWCore.ParameterSet.Config as cms

#------------------------
# SiStripMonitorCluster #
#------------------------

from DQM.SiStripMonitorCluster.SiStripMonitorClusterAlca_cfi import *
SiStripCalZeroBiasMonitorCluster.TopFolderName = cms.string('AlCaReco/SiStrip')

#---------------------------------------------
# Filters to remove APV induced noisy events #
#---------------------------------------------

from DPGAnalysis.SiStripTools.FilterSequenceForAlCaRecoDQM_cfi import *

#------------
# Sequence #
#------------

ALCARECOSiStripCalZeroBiasDQM = cms.Sequence(seqAPVCycleFilter*
                                             SiStripCalZeroBiasMonitorCluster)
# foo bar baz
# OXRIllg1xeKaX
