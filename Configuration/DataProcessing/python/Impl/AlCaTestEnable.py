#!/usr/bin/env python3
"""
_AlCaTestEnable_

Scenario supporting proton collisions

"""

from Configuration.DataProcessing.Impl.AlCa import *

class AlCaTestEnable(AlCa):
    def __init__(self):
        AlCa.__init__(self)
        self.skims=['TkAlLAS']
    """
    _AlCaTestEnable_

    Implement configuration building for data processing for proton
    collision data taking

    """
    def expressProcessing(self, globalTag, **args):
        """
        _expressProcessing_

        Proton collision data taking express processing

        """
        if 'skims' in args:
            if 'EcalTestPulsesRaw' not in args['skims']:
                args['skims'].append('EcalTestPulsesRaw')

        return super(AlCaTestEnable, self).expressProcessing(globalTag, **args)
