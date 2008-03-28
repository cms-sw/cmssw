# The following comments couldn't be translated into the new config version:

# doL1T &
# was like thisl1seedDouble &

# hltL1IsoDoubleElectronL1MatchFilter

#attention Iso
# trackerlocalreco &         

# ckftracks & 

# EgammaHLTRegionalPixelSeedGeneratorProducer

import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1NonIsoDoubleElectronL1MatchFilterRegional = copy.deepcopy(hltEgammaL1MatchFilterRegional)
import copy
from HLTrigger.Egamma.hltEgammaEtFilter_cfi import *
#--------------------------------------------------------------------#
# ET CUT
#--------------------------------------------------------------------#
hltL1NonIsoDoubleElectronEtFilter = copy.deepcopy(hltEgammaEtFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1NonIsoDoubleElectronHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltElectronPixelMatchFilter_cfi import *
#--------------------------------------------------------------------#
# PIXEL MATCH
#--------------------------------------------------------------------#
hltL1NonIsoDoubleElectronPixelMatchFilter = copy.deepcopy(hltElectronPixelMatchFilter)
hltL1NonIsoDoubleElectronL1MatchFilterRegional.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1NonIsoDoubleElectronL1MatchFilterRegional.candNonIsolatedTag = 'l1NonIsoRecoEcalCandidate'
hltL1NonIsoDoubleElectronL1MatchFilterRegional.L1SeedFilterTag = 'l1seedRelaxedDouble'
hltL1NonIsoDoubleElectronL1MatchFilterRegional.ncandcut = 2
hltL1NonIsoDoubleElectronL1MatchFilterRegional.doIsolated = False
hltL1NonIsoDoubleElectronEtFilter.inputTag = 'hltL1NonIsoDoubleElectronL1MatchFilterRegional'
#replace hltL1IsoDoubleElectronEtFilter.inputTag = hltL1IsoDoubleElectronL1MatchFilter
hltL1NonIsoDoubleElectronEtFilter.etcut = 19.0
hltL1NonIsoDoubleElectronEtFilter.ncandcut = 2
hltL1NonIsoDoubleElectronHcalIsolFilter.candTag = 'hltL1NonIsoDoubleElectronEtFilter'
hltL1NonIsoDoubleElectronHcalIsolFilter.isoTag = 'l1IsolatedElectronHcalIsol'
hltL1NonIsoDoubleElectronHcalIsolFilter.nonIsoTag = 'l1NonIsolatedElectronHcalIsol'
hltL1NonIsoDoubleElectronHcalIsolFilter.hcalisolbarrelcut = 9.
hltL1NonIsoDoubleElectronHcalIsolFilter.hcalisolendcapcut = 9.
hltL1NonIsoDoubleElectronHcalIsolFilter.ncandcut = 2
hltL1NonIsoDoubleElectronHcalIsolFilter.doIsolated = False
hltL1NonIsoDoubleElectronPixelMatchFilter.candTag = 'hltL1NonIsoDoubleElectronHcalIsolFilter'
hltL1NonIsoDoubleElectronPixelMatchFilter.L1IsoPixelSeedsTag = 'l1IsoElectronPixelSeeds'

