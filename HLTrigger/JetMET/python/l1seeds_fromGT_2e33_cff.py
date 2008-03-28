import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
# L1 Seed filters
l1s1jet = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s2jet = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s3jet = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s4jet = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s1MET = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s1HT = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s1jet1MET = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s2jetAco = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s1jet1METAco = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s2jet1MET = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s3jet1MET = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s4jet1MET = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s2jetvbfMET = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1snvMET = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1sPhi2MET = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s1MET1HT = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s1jetPE1 = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s1jetPE3 = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s1jetPE5 = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1RapGap = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1sdijetave60 = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1sdijetave110 = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1sdijetave150 = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1sdijetave200 = copy.deepcopy(hltLevel1GTSeed)
l1s1jet.L1SeedsLogicalExpression = 'L1_SingleJet200'
l1s2jet.L1SeedsLogicalExpression = 'L1_SingleJet200 OR L1_DoubleJet100'
l1s3jet.L1SeedsLogicalExpression = 'L1_SingleJet200 OR L1_DoubleJet100 OR L1_TripleJet50'
l1s4jet.L1SeedsLogicalExpression = 'L1_SingleJet200 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet40'
l1s1MET.L1SeedsLogicalExpression = 'L1_ETM60'
l1s1HT.L1SeedsLogicalExpression = 'L1_HTT300'
l1s1jet1MET.L1SeedsLogicalExpression = 'L1_ETM60'
l1s2jetAco.L1SeedsLogicalExpression = 'L1_SingleJet200 OR L1_DoubleJet100'
l1s1jet1METAco.L1SeedsLogicalExpression = 'L1_ETM60'
l1s2jet1MET.L1SeedsLogicalExpression = 'L1_ETM60'
l1s3jet1MET.L1SeedsLogicalExpression = 'L1_ETM60'
l1s4jet1MET.L1SeedsLogicalExpression = 'L1_ETM60'
l1s2jetvbfMET.L1SeedsLogicalExpression = 'L1_ETM60'
l1snvMET.L1SeedsLogicalExpression = 'L1_ETM60'
l1sPhi2MET.L1SeedsLogicalExpression = 'L1_ETM60'
l1s1MET1HT.L1SeedsLogicalExpression = 'L1_HTT300'
l1s1jetPE1.L1SeedsLogicalExpression = 'L1_SingleJet150'
l1s1jetPE3.L1SeedsLogicalExpression = 'L1_SingleJet50'
l1s1jetPE5.L1SeedsLogicalExpression = 'L1_SingleJet20'
l1RapGap.L1SeedsLogicalExpression = 'L1_IsoEG15_Jet20_ForJet10'
l1sdijetave60.L1SeedsLogicalExpression = 'L1_HTT100'
l1sdijetave110.L1SeedsLogicalExpression = 'L1_HTT200'
l1sdijetave150.L1SeedsLogicalExpression = 'L1_HTT250'
l1sdijetave200.L1SeedsLogicalExpression = 'L1_HTT300'

