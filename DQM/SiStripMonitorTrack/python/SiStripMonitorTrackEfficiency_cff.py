import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorTrack.SiStripMonitorTrackEfficiency_cfi import *
#-----------------------
#  Reconstruction Modules
#-----------------------
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
DQMsiStripMonitorTrackEfficiency_Sim = cms.Sequence(siStripDigis*siStripZeroSuppression*cms.SequencePlaceholder("TrackRefitter")*siStripMonitorTrackEfficiency)
DQMSiStripMonitorTrackEfficiency_Real = cms.Sequence(siStripMonitorTrackEfficiency)
siStripMonitorTrackEfficiency.TrackProducer = 'ctfWithMaterialTracksP5'
siStripMonitorTrackEfficiency.TrackLabel = ''
siStripMonitorTrackEfficiency.OutputMEsInRootFile = True
siStripMonitorTrackEfficiency.OutputFileName = '/tmp/giordano/sistripmonitortrack_test21X.root'
siStripMonitorTrackEfficiency.Cluster_src = 'siStripClusters'
siStripMonitorTrackEfficiency.Mod_On = False
siStripMonitorTrackEfficiency.OffHisto_On = True ##false


