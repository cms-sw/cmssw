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
    diamondSampicLocalReconstructionTask ,
    ctppsPixelLocalReconstructionTask ,
    ctppsLocalTrackLiteProducer ,
    ctppsProtons
)

from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018
ctpps_2018.toReplaceWith(
    recoCTPPSTask,
    cms.Task(
        totemRPLocalReconstructionTask ,
        ctppsDiamondLocalReconstructionTask ,
        totemTimingLocalReconstructionTask ,
        ctppsPixelLocalReconstructionTask ,
        ctppsLocalTrackLiteProducer ,
        ctppsProtons
    )
)

recoCTPPS = cms.Sequence(recoCTPPSTask)

# reconstruction for direct simulation (RecHit-level info)

totemRPFromRHLocalReconstructionTask = totemRPLocalReconstructionTask.copyAndExclude([totemRPClusterProducer, totemRPRecHitProducer])
ctppsDiamondFromRHLocalReconstructionTask = ctppsDiamondLocalReconstructionTask.copyAndExclude([ctppsDiamondRecHits])
ctppsPixelFromRHLocalReconstructionTask = ctppsPixelLocalReconstructionTask.copyAndExclude([ctppsPixelClusters, ctppsPixelRecHits])

recoDirectSimPPSTask = cms.Task(
    totemRPFromRHLocalReconstructionTask,
    ctppsDiamondFromRHLocalReconstructionTask,
    ctppsPixelFromRHLocalReconstructionTask,
    ctppsLocalTrackLiteProducer,
    ctppsProtons
)

recoDirectSimPPS = cms.Sequence(recoDirectSimPPSTask)
