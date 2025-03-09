#!/usr/bin/env python3
"""
_hltScoutingEra_Run3_2025_

Scenario supporting proton collisions with input HLT scouting data for 2025

"""

import os
import sys

from Configuration.Eras.Era_Run3_2025_cff import Run3_2025
from Configuration.DataProcessing.Impl.hltScouting import hltScouting

class hltScoutingEra_Run3_2025(hltScouting):
    def __init__(self):
        hltScouting.__init__(self)
        self.recoSeq = ''
        self.cbSc = 'pp'
        self.eras = Run3_2025
        self.promptCustoms += ['Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2025']
    """
    _hltScoutingEra_Run3_2025_
    Implement configuration building for data processing for proton
    collision data taking with input HLT scouting data for Era_Run3_2025
    """
