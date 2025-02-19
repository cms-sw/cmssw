import FWCore.ParameterSet.Config as cms

import warnings
warnings.warn('Configuration.StandardSequences.GeometryTrackerOnly_cff is deprecated, please use Configuration.Geometry.GeometryTrackerOnly_cff', DeprecationWarning, stacklevel=2)

from Configuration.Geometry.GeometryTrackerOnly_cff import *
