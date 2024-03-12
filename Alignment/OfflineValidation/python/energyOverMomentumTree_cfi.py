import FWCore.ParameterSet.Config as cms

from Alignment.OfflineValidation.eopTreeWriter_cfi import eopTreeWriter as _eopTreeWriter
energyOverMomentumTree = _eopTreeWriter.clone(
    src = 'TrackRefitter'
)
# foo bar baz
# VvFi61QWuF10I
# UMWkH8b4qJ4Yv
