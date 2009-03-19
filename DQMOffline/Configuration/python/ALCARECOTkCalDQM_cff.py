#
# Define which AlCa streams are to be monitored for what
# Tk Calibration is concerned
#
# $Id$
#

# SiStripMonitorCluster #
import DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi

ALCARECOSiStripCalZeroBiasDQM = DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi.SiStripMonitorCluster.clone()

ALCARECOSiStripCalZeroBiasDQM.OutputMEsInRootFile                     = False 
ALCARECOSiStripCalZeroBiasDQM.SelectAllDetectors                      = True 
ALCARECOSiStripCalZeroBiasDQM.StripQualityLabel                       = 'unbiased'

ALCARECOSiStripCalZeroBiasDQM.TH1ClusterPos.moduleswitchon            = True
ALCARECOSiStripCalZeroBiasDQM.TH1ClusterPos.layerswitchon             = False
ALCARECOSiStripCalZeroBiasDQM.TH1nClusters.moduleswitchon             = False
ALCARECOSiStripCalZeroBiasDQM.TH1nClusters.layerswitchon              = False
ALCARECOSiStripCalZeroBiasDQM.TH1ClusterStoN.moduleswitchon           = False
ALCARECOSiStripCalZeroBiasDQM.TH1ClusterStoN.layerswitchon            = False
ALCARECOSiStripCalZeroBiasDQM.TH1ClusterStoNVsPos.moduleswitchon      = False
ALCARECOSiStripCalZeroBiasDQM.TH1ClusterStoNVsPos.layerswitchon       = False
ALCARECOSiStripCalZeroBiasDQM.TH1ClusterNoise.moduleswitchon          = False
ALCARECOSiStripCalZeroBiasDQM.TH1ClusterNoise.layerswitchon           = False
ALCARECOSiStripCalZeroBiasDQM.TH1NrOfClusterizedStrips.moduleswitchon = False
ALCARECOSiStripCalZeroBiasDQM.TH1NrOfClusterizedStrips.layerswitchon  = False
ALCARECOSiStripCalZeroBiasDQM.TH1ModuleLocalOccupancy.moduleswitchon  = False
ALCARECOSiStripCalZeroBiasDQM.TH1ModuleLocalOccupancy.layerswitchon   = False
ALCARECOSiStripCalZeroBiasDQM.TH1ClusterCharge.moduleswitchon         = False
ALCARECOSiStripCalZeroBiasDQM.TH1ClusterCharge.layerswitchon          = False
ALCARECOSiStripCalZeroBiasDQM.TH1ClusterWidth.moduleswitchon          = False
ALCARECOSiStripCalZeroBiasDQM.TH1ClusterWidth.layerswitchon           = False
ALCARECOSiStripCalZeroBiasDQM.TProfNumberOfCluster.moduleswitchon     = False
ALCARECOSiStripCalZeroBiasDQM.TProfNumberOfCluster.layerswitchon      = False
ALCARECOSiStripCalZeroBiasDQM.TProfClusterWidth.moduleswitchon        = False
ALCARECOSiStripCalZeroBiasDQM.TProfClusterWidth.layerswitchon         = False

