from PhysicsTools.HeppyCore.particles.met import MET as BaseMET

class MET(BaseMET):
    def __init__(self, tlv, sum_et):
        self._tlv = tlv
        self._sum_et = sum_et
