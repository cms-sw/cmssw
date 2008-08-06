import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorTrack.SiStripMonitorTrackEfficiency_cfi import *

SiStripMonitorTrackEfficiency.TrackProducer = 'ctfWithMaterialTracksP5'
SiStripMonitorTrackEfficiency.TrackLabel    = ''
SiStripMonitorTrackEfficiency.OutputMEsInRootFile = True
SiStripMonitorTrackEfficiency.OutputFileName= '/tmp/giordano/sistripmonitortrack_test21X.root'
SiStripMonitorTrackEfficiency.Cluster_src = 'siStripClusters'
SiStripMonitorTrackEfficiency.Mod_On      = False
SiStripMonitorTrackEfficiency.OffHisto_On = True


#-----------------------
#  Reconstruction Modules
#-----------------------
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *

from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *

DQMsiStripMonitorTrackEfficiency_Sim = cms.Sequence( siStripDigis
                                                     *
                                                     siStripZeroSuppression
                                                     *cms.SequencePlaceholder("TrackRefitter")
                                                     *
                                                     SiStripMonitorTrackEfficiency
                                                     )

DQMSiStripMonitorTrackEfficiency_Real = cms.Sequence( SiStripMonitorTrackEfficiency
                                                      )
