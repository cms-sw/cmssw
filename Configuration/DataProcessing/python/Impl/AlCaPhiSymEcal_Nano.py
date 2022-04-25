#!/usr/bin/env python3
"""
_AlCaPhiSymEcal_Nano_

Scenario supporting proton collision data taking for AlCaPhiSymEcal stream with ALCANANO output

"""

from Configuration.DataProcessing.Impl.AlCa import AlCa
from Configuration.Eras.Era_Run3_cff import Run3

class AlCaPhiSymEcal_Nano(AlCa):
    def __init__(self):
        AlCa.__init__(self)
        self.skims=['EcalPhiSymByRun']
        self.eras=Run3
        self.promptCustoms = [ 'Calibration/EcalCalibAlgos/EcalPhiSymRecoSequence_cff.customise' ]
        self.step = 'RECO:bunchSpacingProducer+ecalMultiFitUncalibRecHitTask+ecalCalibratedRecHitTask'
    """
    _AlCaPhiSymEcal_Nano_

    Implement configuration building for data processing for proton
    collision data taking for AlCaPhiSymEcal stream with ALCANANO output

    """
