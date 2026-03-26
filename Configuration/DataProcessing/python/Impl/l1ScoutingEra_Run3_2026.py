#!/usr/bin/env python3
"""
_l1ScoutingEra_Run3_2026_

Scenario supporting proton collisions with input L1-Scouting data for 2026

"""
from Configuration.DataProcessing.Impl.l1Scouting import l1Scouting

from Configuration.Eras.Era_Run3_2026_cff import Run3_2026

class l1ScoutingEra_Run3_2026(l1Scouting):
    def __init__(self):
        l1Scouting.__init__(self)
        self.recoSeq = ''
        self.cbSc = 'pp'
        self.eras = Run3_2026
        self.promptCustoms += ['Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2026']
    """
    _l1ScoutingEra_Run3_2026_
    Implement configuration building for data processing for proton
    collision data taking with input L1-Scouting data for Era_Run3_2026
    """
