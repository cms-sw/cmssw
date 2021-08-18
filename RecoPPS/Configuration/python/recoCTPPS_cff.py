import FWCore.ParameterSet.Config as cms

from RecoPPS.Local.totemRPLocalReconstruction_cff import *
from RecoPPS.Local.ctppsDiamondLocalReconstruction_cff import *
from RecoPPS.Local.totemTimingLocalReconstruction_cff import *
from RecoPPS.Local.ctppsPixelLocalReconstruction_cff import *

from RecoPPS.Local.ctppsLocalTrackLiteProducer_cff import ctppsLocalTrackLiteProducer

from RecoPPS.ProtonReconstruction.ctppsProtons_cff import *

from Geometry.VeryForwardGeometry.geometryRPFromDB_cfi import *
from CalibPPS.ESProducers.ppsTopology_cff import *

recoCTPPSTask = cms.Task(
    totemRPLocalReconstructionTask ,
    ctppsDiamondLocalReconstructionTask ,
    totemTimingLocalReconstructionTask ,
    ctppsPixelLocalReconstructionTask ,
    ctppsLocalTrackLiteProducer ,
    ctppsProtons
)

recoCTPPS = cms.Sequence(recoCTPPSTask)
