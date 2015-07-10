import FWCore.ParameterSet.Config as cms
import copy

from CMGTools.H2TauTau.eventContent.common_cff import common, commonDebug

tauMuSpecific = [
    ]

tauMuSpecificDebug = copy.deepcopy(tauMuSpecific)
tauMuSpecificDebug.extend([
    ])

tauMu = copy.deepcopy(common)
tauMu.extend(tauMuSpecific)

tauMuDebug = copy.deepcopy(commonDebug)
tauMuDebug.extend(tauMuSpecificDebug)

