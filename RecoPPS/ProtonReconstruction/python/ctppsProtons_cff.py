import FWCore.ParameterSet.Config as cms

# import default alignment settings
from CalibPPS.ESProducers.ctppsAlignment_cff import *

# import default optics settings
from CalibPPS.ESProducers.ctppsOpticalFunctions_cff import *

# import and adjust proton-reconstructions settings
from RecoPPS.ProtonReconstruction.ctppsProtons_cfi import *
from CalibPPS.ESProducers.ppsAssociationCuts_non_DB_cff import *

ctppsProtons.lhcInfoLabel = ctppsLHCInfoLabel
ctppsProtons.ppsAssociationCutsLabel = ppsAssociationCutsESSource.ppsAssociationCutsLabel

ctppsProtons.pixelDiscardBXShiftedTracks = True
ctppsProtons.default_time = -999.
