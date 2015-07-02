import FWCore.ParameterSet.Config as cms
import copy

from CMGTools.H2TauTau.eventContent.common_cff import common, commonDebug

tauEleSpecific = [
    ]

tauEleSpecificDebug = copy.deepcopy(tauEleSpecific)
tauEleSpecificDebug.extend([
    ])

tauEle = copy.deepcopy(common)
tauEle.extend(tauEleSpecific)

tauEleDebug = copy.deepcopy(commonDebug)
tauEleDebug.extend(tauEleSpecificDebug)

