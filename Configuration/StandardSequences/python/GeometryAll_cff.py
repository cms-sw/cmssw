import FWCore.ParameterSet.Config as cms

import warnings
warnings.warn('Configuration.StandardSequences.GeometryAll_cff is deprecated, please use Configuration.Geometry.GeometryAll_cff', DeprecationWarning, stacklevel=2)

from Configuration.Geometry.GeometryAll_cff import *
