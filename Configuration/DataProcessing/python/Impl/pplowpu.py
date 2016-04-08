#!/usr/bin/env python
"""
_pplowpu_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Impl.pp import pp

class pplowpu(pp):
    def __init__(self):
        pp.__init__(self)
        self.cbSc='pp'
        cCustoms = [ 'RecoTracker/Configuration/customiseForRunI.customiseForRunI' ]
        self.promptCustoms += cCustoms
        self.expressCustoms += cCustoms
        self.visCustoms += cCustoms
    """
    _pplowpu_

    Implement configuration building for data processing for proton
    collision data taking

    """
