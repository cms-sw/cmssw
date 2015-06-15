import FWCore.ParameterSet.Config as cms
import copy

from CMGTools.H2TauTau.eventContent.common_cff import common, commonDebug

muEleSpecific = [
    ]

muEleSpecificDebug = copy.deepcopy(muEleSpecific)
muEleSpecificDebug.extend([
    ])

muEle = copy.deepcopy(common)
muEle.extend(muEleSpecific)

muEleDebug = copy.deepcopy(commonDebug)
muEleDebug.extend(muEleSpecificDebug)

