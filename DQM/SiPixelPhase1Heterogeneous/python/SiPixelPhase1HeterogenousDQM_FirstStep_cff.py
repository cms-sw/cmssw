import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Heterogeneous.SiPixelPhase1MonitorTrackSoA_cfi import *
from DQM.SiPixelPhase1Heterogeneous.SiPixelPhase1MonitorVertexSoA_cfi import *
#from DQM.SiPixelPhase1Heterogeneous.SiPixelPhase1MonitorDigiSoA_cfi import *

monitorpixelsoa = cms.Task(monitorpixelDigi,
                           monitorpixelTrackSoA)
#,
#                           monitorpixelVertexSoA,
#                       )
