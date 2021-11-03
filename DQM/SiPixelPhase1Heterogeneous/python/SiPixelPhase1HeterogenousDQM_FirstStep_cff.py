import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Heterogeneous.monitorpixelTrackSoA_cfi import *
from DQM.SiPixelPhase1Heterogeneous.monitorpixelVertexSoA_cfi import *

monitorpixelSoASource = cms.Sequence(monitorpixelTrackSoA * monitorpixelVertexSoA)
