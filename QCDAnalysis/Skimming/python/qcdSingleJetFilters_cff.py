import FWCore.ParameterSet.Config as cms

import copy
from QCDAnalysis.Skimming.qcdSingleJetFilter_cfi import *
qcdSingleJetFilterHi = copy.deepcopy(qcdSingleJetFilter)
import copy
from QCDAnalysis.Skimming.qcdSingleJetFilter_cfi import *
qcdSingleJetFilterMed = copy.deepcopy(qcdSingleJetFilter)
import copy
from QCDAnalysis.Skimming.qcdSingleJetFilter_cfi import *
qcdSingleJetFilterLo = copy.deepcopy(qcdSingleJetFilter)
qcdSingleJetFilterHi.MinPt = 1000
qcdSingleJetFilterMed.MinPt = 500
qcdSingleJetFilterLo.MinPt = 200

