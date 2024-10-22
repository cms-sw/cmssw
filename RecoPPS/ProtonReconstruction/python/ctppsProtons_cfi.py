# import and adjust proton-reconstructions settings
from RecoPPS.ProtonReconstruction.ctppsProtonsDefault_cfi import ctppsProtonsDefault as _ctppsProtonsDefault
ctppsProtons = _ctppsProtonsDefault.clone(
     pixelDiscardBXShiftedTracks = True,
     default_time = -999.
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(ctppsProtons, useNewLHCInfo = True)

from Configuration.Eras.Modifier_ctpps_directSim_cff import ctpps_directSim
ctpps_directSim.toModify(ctppsProtons, useNewLHCInfo = False)
