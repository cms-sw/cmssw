import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.SimL1EmulatorDM_cff import *

for _entry in [SimL1TechnicalTriggers]:
    SimL1Emulator.remove(_entry)

del simGtDigis.TechnicalTriggersInputTags

import FastSimulation.Configuration.DigiAndMixAliasInfo_cff as _aliasInfo
gtDigis = _aliasInfo.infoToAlias(_aliasInfo.gtDigisAliasInfo)
gmtDigis = _aliasInfo.infoToAlias(_aliasInfo.gmtDigisAliasInfo)
