import FWCore.ParameterSet.Config as cms

class Eras (object):
    """
    Dummy container for all the cms.Modifier instances that config fragments
    can use to selectively configure depending on what scenario is active.
    """
    def __init__(self):
        self.run2 = cms.Modifier()
        self.bunchspacing25ns = cms.Modifier()
        self.bunchspacing50ns = cms.Modifier()

eras=Eras()
