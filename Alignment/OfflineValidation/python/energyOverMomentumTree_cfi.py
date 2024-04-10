import FWCore.ParameterSet.Config as cms

from Alignment.OfflineValidation.eopTreeWriter_cfi import eopTreeWriter as _eopTreeWriter
energyOverMomentumTree = _eopTreeWriter.clone(
    src = 'TrackRefitter'
)
