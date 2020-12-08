import FWCore.ParameterSet.Config as cms

from RecoPPS.Local.totemRPLocalReconstruction_cff import *
from RecoPPS.Local.ctppsDiamondLocalReconstruction_cff import *
from RecoPPS.Local.totemTimingLocalReconstruction_cff import *
from RecoPPS.Local.ctppsPixelLocalReconstruction_cff import *

from RecoPPS.Local.ctppsLocalTrackLiteProducer_cff import ctppsLocalTrackLiteProducer

from RecoPPS.ProtonReconstruction.ctppsProtons_cff import *

from Geometry.VeryForwardGeometry.geometryRPFromDB_cfi import *

recoCTPPSTask = cms.Task(
    totemRPLocalReconstructionTask ,
    ctppsDiamondLocalReconstructionTask ,
    totemTimingLocalReconstructionTask ,
    ctppsPixelLocalReconstructionTask ,
    ctppsLocalTrackLiteProducer ,
    ctppsProtons
)

#temporarily remove ctppsProtons in Run-3 (see issue #32340)
from Configuration.Eras.Modifier_ctpps_2021_cff import ctpps_2021
_ctpps_2021_recoCTPPSTask = recoCTPPSTask.copyAndExclude([ctppsProtons])
ctpps_2021.toReplaceWith(recoCTPPSTask, _ctpps_2021_recoCTPPSTask)

recoCTPPS = cms.Sequence(recoCTPPSTask)
