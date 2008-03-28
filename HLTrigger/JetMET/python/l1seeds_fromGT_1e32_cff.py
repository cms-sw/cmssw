import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
# L1 Seed filters
l1s1jet = copy.deepcopy(hltLevel1GTSeed)
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
l1s1jetPE7 = copy.deepcopy(hltLevel1GTSeed)
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
l1s1METPre1 = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s1METPre2 = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s1METPre3 = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s1HT = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s1SumET = copy.deepcopy(hltLevel1GTSeed)
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
l1sPhiJet1MET = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1sPhiJet2MET = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1sPhiJet1Jet2 = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1s1MET1HT = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1RapGap = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1sdijetave30 = copy.deepcopy(hltLevel1GTSeed)
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
l1s1jet.L1SeedsLogicalExpression = 'L1_SingleJet150'
l1s1jetPE1.L1SeedsLogicalExpression = 'L1_SingleJet100'
l1s1jetPE3.L1SeedsLogicalExpression = 'L1_SingleJet70'
l1s1jetPE5.L1SeedsLogicalExpression = 'L1_SingleJet30'
l1s1jetPE7.L1SeedsLogicalExpression = 'L1_SingleJet15'
l1s2jet.L1SeedsLogicalExpression = 'L1_SingleJet150 OR L1_DoubleJet70'
l1s3jet.L1SeedsLogicalExpression = 'L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50'
l1s4jet.L1SeedsLogicalExpression = 'L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet40'
l1s1MET.L1SeedsLogicalExpression = 'L1_ETM40'
l1s1METPre1.L1SeedsLogicalExpression = 'L1_ETM40'
l1s1METPre2.L1SeedsLogicalExpression = 'L1_ETM15'
l1s1METPre3.L1SeedsLogicalExpression = 'L1_ETM10'
l1s1HT.L1SeedsLogicalExpression = 'L1_HTT300'
l1s1SumET.L1SeedsLogicalExpression = 'L1_ETT60'
l1s1jet1MET.L1SeedsLogicalExpression = 'L1_SingleJet150'
l1s2jetAco.L1SeedsLogicalExpression = 'L1_SingleJet150 OR L1_DoubleJet70'
l1s1jet1METAco.L1SeedsLogicalExpression = 'L1_SingleJet150'
l1s2jet1MET.L1SeedsLogicalExpression = 'L1_SingleJet150'
l1s3jet1MET.L1SeedsLogicalExpression = 'L1_SingleJet150'
l1s4jet1MET.L1SeedsLogicalExpression = 'L1_SingleJet150'
l1s2jetvbfMET.L1SeedsLogicalExpression = 'L1_ETM40'
l1snvMET.L1SeedsLogicalExpression = 'L1_SingleJet150'
l1sPhi2MET.L1SeedsLogicalExpression = 'L1_SingleJet150'
l1sPhiJet1MET.L1SeedsLogicalExpression = 'L1_SingleJet150'
l1sPhiJet2MET.L1SeedsLogicalExpression = 'L1_SingleJet150'
l1sPhiJet1Jet2.L1SeedsLogicalExpression = 'L1_SingleJet150'
l1s1MET1HT.L1SeedsLogicalExpression = 'L1_HTT300'
l1RapGap.L1SeedsLogicalExpression = 'L1_IsoEG10_Jet15_ForJet10'
l1sdijetave30.L1SeedsLogicalExpression = 'L1_SingleJet15'
l1sdijetave60.L1SeedsLogicalExpression = 'L1_SingleJet30'
l1sdijetave110.L1SeedsLogicalExpression = 'L1_SingleJet70'
l1sdijetave150.L1SeedsLogicalExpression = 'L1_SingleJet100'
l1sdijetave200.L1SeedsLogicalExpression = 'L1_SingleJet150'

