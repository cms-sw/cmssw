import FWCore.ParameterSet.Config as cms

from Geometry.MTDGeometryBuilder.mtdParameters_cfi import (
    mtdParameters as _mtdParameters,
)

hltPhase2mtdParameters = _mtdParameters.clone()
