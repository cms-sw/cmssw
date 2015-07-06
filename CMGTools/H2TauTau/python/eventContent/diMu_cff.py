import FWCore.ParameterSet.Config as cms
import copy

from CMGTools.H2TauTau.eventContent.common_cff import common, commonDebug

diMuSpecific = [
    ]

diMuSpecificDebug = copy.deepcopy(diMuSpecific)
diMuSpecificDebug.extend([
    ])

diMu = copy.deepcopy(common)
diMu.extend(diMuSpecific)

diMuDebug = copy.deepcopy(commonDebug)
diMuDebug.extend(diMuSpecificDebug)

