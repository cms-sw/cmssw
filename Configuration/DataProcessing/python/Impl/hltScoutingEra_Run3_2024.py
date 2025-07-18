#!/usr/bin/env python3
"""
_hltScoutingEra_Run3_2024_

Scenario supporting proton collisions with input HLT scouting data for 2024

"""

import os
import sys

from Configuration.Eras.Era_Run3_2024_cff import Run3_2024
from Configuration.DataProcessing.Impl.hltScouting import hltScouting

class hltScoutingEra_Run3_2024(hltScouting):
    def __init__(self):
        hltScouting.__init__(self)
        self.recoSeq = ''
        self.cbSc = 'pp'
        self.eras = Run3_2024
        self.promptCustoms += ['Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2024']
    """
    _hltScoutingEra_Run3_2024_
    Implement configuration building for data processing for proton
    collision data taking with input HLT scouting data for Era_Run3_2024
    """
