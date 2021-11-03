import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1MonitorTrackSoA_cfi import *
from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1MonitorVertexSoA_cfi import *

monitorpixelSoASource = cms.Sequence(siPixelPhase1MonitorTrackSoA * siPixelPhase1MonitorVertexSoA)
