import FWCore.ParameterSet.Config as cms
import copy

from CMGTools.H2TauTau.eventContent.common_cff import common, commonDebug 

diTauSpecific = [
    ]

diTauSpecificDebug = copy.deepcopy( diTauSpecific)
diTauSpecificDebug.extend( [
    ] )

diTau = copy.deepcopy( common )
diTau.extend(diTauSpecific)

diTauDebug = copy.deepcopy( commonDebug )
diTauDebug.extend( diTauSpecificDebug )
