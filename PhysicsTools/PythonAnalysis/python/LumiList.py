#!/usr/bin/env python

from FWCore.PythonUtilities.LumiList import LumiList as ll
import warnings

warnings.warn('PhysicsTools.PythonAnalysis.LumiList is deprecated, please use FWCore.PythonUtilities.LumiList', DeprecationWarning, stacklevel=2)

class LumiList(ll):
    pass
