import FWCore.ParameterSet.Config as cms

#include "DQM/HLTEvF/data/HLTMonElectron.cfi"
from DQM.HLTEvF.doubleElectronDQM_cfi import *
from DQM.HLTEvF.doubleElectronRelaxedDQM_cfi import *
from DQM.HLTEvF.doublePhotonDQM_cfi import *
from DQM.HLTEvF.doublePhotonRelaxedDQM_cfi import *
from DQM.HLTEvF.highEtDQM_cfi import *
from DQM.HLTEvF.singleElectronDQM_cfi import *
from DQM.HLTEvF.singleElectronLargeWindowDQM_cfi import *
from DQM.HLTEvF.singleElectronRelaxedDQM_cfi import *
from DQM.HLTEvF.singleElectronRelaxedLargeWindowDQM_cfi import *
from DQM.HLTEvF.singlePhotonDQM_cfi import *
from DQM.HLTEvF.singlePhotonRelaxedDQM_cfi import *
from DQM.HLTEvF.veryHighEtDQM_cfi import *
hltMonElectronPath = cms.Path(singleElectronRelaxedDQM+singleElectronDQM+singlePhotonRelaxedDQM+singlePhotonDQM+doubleElectronRelaxedDQM+doubleElectronDQM+doublePhotonRelaxedDQM+doublePhotonDQM+highEtDQM+veryHighEtDQM)

