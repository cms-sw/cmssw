from math import *

class EmbeddedElectronCalibrator:
    def __init__(self,label="calib"):
        self._label = label
    def correct(self,cmgelectron,dummy):
        ele = cmgelectron.sourcePtr().get()
        if not ele.hasUserFloat("p_"+self._label): 
            raise RuntimeError, "Electron does not have an embedded energy scale correction with label '%s'" % self._label
        kind_in = ele.candidateP4Kind()
        p4_in = ele.p4(kind_in)
        pCalib    = ele.userFloat("p_"+self._label)
        pErrCalib = ele.userFloat("pError_"+self._label)
        pKindCalib = ele.userInt("pKind_"+self._label)
        ecalCalib  = ele.userFloat("ecalEnergy_"+self._label)
        eErrCalib  = ele.userFloat("ecalEnergyError_"+self._label)
        ele.setCorrectedEcalEnergy( ecalCalib )
        ele.setCorrectedEcalEnergyError( eErrCalib )
        p4_out = p4_in * (pCalib/p4_in.P())
        ele.setP4(pKindCalib, p4_out, pErrCalib, True)
        cmgelectron.setP4(p4_out)

