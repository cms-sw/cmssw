import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
#JetMET filters to be used in JetMET HLT paths for luminosity L=10^32 cm-2s-1
# Single jet triggers
hlt1jet30 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt1jet50 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt1jet60 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt1jet100 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt1jet110 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt1jet120 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt1jet150 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt1jet180 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt1jet200 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt1jet250 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt1jet400 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
# Double jet triggers
hlt2jet125 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt2jet150 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt2jet155 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt2jet200 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt2jet350 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.JetMET.hltJetVBFFilter_cfi import *
#module hlt2jetvbf = HLTJetVBFFilter {
#   InputTag inputTag = iterativeCone5CaloJets
#   double minEt = 40.0
#   double minDeltaEta = 4.2	   
#}
hlt2jetvbf = copy.deepcopy(hltJetVBFFilter)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
# Triple jet triggers
hlt3jet60 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt3jet75 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt3jet85 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt3jet100 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt3jet195 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
# Quadruple jet triggers
hlt4jet35 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt4jet50 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt4jet60 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt4jet80 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloMET_cfi import *
# MET
hlt1MET15 = copy.deepcopy(hlt1CaloMET)
import copy
from HLTrigger.HLTfilters.hlt1CaloMET_cfi import *
hlt1MET20 = copy.deepcopy(hlt1CaloMET)
import copy
from HLTrigger.HLTfilters.hlt1CaloMET_cfi import *
hlt1MET30 = copy.deepcopy(hlt1CaloMET)
import copy
from HLTrigger.HLTfilters.hlt1CaloMET_cfi import *
hlt1MET35 = copy.deepcopy(hlt1CaloMET)
import copy
from HLTrigger.HLTfilters.hlt1CaloMET_cfi import *
hlt1MET40 = copy.deepcopy(hlt1CaloMET)
import copy
from HLTrigger.HLTfilters.hlt1CaloMET_cfi import *
hlt1MET55 = copy.deepcopy(hlt1CaloMET)
import copy
from HLTrigger.HLTfilters.hlt1CaloMET_cfi import *
hlt1MET60 = copy.deepcopy(hlt1CaloMET)
import copy
from HLTrigger.HLTfilters.hlt1CaloMET_cfi import *
hlt1MET65 = copy.deepcopy(hlt1CaloMET)
import copy
from HLTrigger.HLTfilters.hlt1CaloMET_cfi import *
hlt1MET70 = copy.deepcopy(hlt1CaloMET)
import copy
from HLTrigger.HLTfilters.hlt1CaloMET_cfi import *
hlt1MET75 = copy.deepcopy(hlt1CaloMET)
import copy
from HLTrigger.HLTfilters.hlt1CaloMET_cfi import *
hlt1MET80 = copy.deepcopy(hlt1CaloMET)
import copy
from HLTrigger.HLTfilters.hlt1CaloMET_cfi import *
hlt1MET91 = copy.deepcopy(hlt1CaloMET)
import copy
from HLTrigger.JetMET.hltRapGapFilter_cfi import *
#Diffractive triggers
hltRapGap = copy.deepcopy(hltRapGapFilter)
import copy
from HLTrigger.JetMET.hltDiJetAveFilter_cfi import *
# Dijet balance triggers
hltdijetave30 = copy.deepcopy(hltDiJetAveFilter)
import copy
from HLTrigger.JetMET.hltDiJetAveFilter_cfi import *
hltdijetave60 = copy.deepcopy(hltDiJetAveFilter)
import copy
from HLTrigger.JetMET.hltDiJetAveFilter_cfi import *
hltdijetave110 = copy.deepcopy(hltDiJetAveFilter)
import copy
from HLTrigger.JetMET.hltDiJetAveFilter_cfi import *
hltdijetave150 = copy.deepcopy(hltDiJetAveFilter)
import copy
from HLTrigger.JetMET.hltDiJetAveFilter_cfi import *
hltdijetave200 = copy.deepcopy(hltDiJetAveFilter)
import copy
from HLTrigger.JetMET.hltNVFilter_cfi import *
# New SUSY filters 
hltnv = copy.deepcopy(hltNVFilter)
import copy
from HLTrigger.JetMET.hltPhi2METFilter_cfi import *
hltPhi2metAco = copy.deepcopy(hltPhi2METFilter)
import copy
from HLTrigger.JetMET.hltPhiJet1METFilter_cfi import *
hltPhiJet1metAco = copy.deepcopy(hltPhiJet1METFilter)
import copy
from HLTrigger.JetMET.hltPhiJet2METFilter_cfi import *
hltPhiJet2metAco = copy.deepcopy(hltPhiJet2METFilter)
import copy
from HLTrigger.JetMET.hltPhiJet1Jet2Filter_cfi import *
hltPhiJet1Jet2Aco = copy.deepcopy(hltPhiJet1Jet2Filter)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
# HLTPrescaler module in each path (for dynamic prescale changes)
pre1jet = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre2jet = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre3jet = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre4jet = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre1MET = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre1METPre1 = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre1METPre2 = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre1METPre3 = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre1HT = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre1jet1MET = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre2jetAco = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre1jet1METAco = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre2jet1MET = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre3jet1MET = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre4jet1MET = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre2jetvbfMET = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prenv = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prephijet1met = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prephijet2met = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prephijet1jet2 = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prephi2met = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre1MET1HT = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre1MET1SumET = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre1jetPE1 = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre1jetPE3 = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre1jetPE5 = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
pre1jetPE7 = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prerapgap = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
predijetave30 = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
predijetave60 = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
predijetave110 = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
predijetave150 = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
predijetave200 = copy.deepcopy(hltPrescaler)
# old MET HT using met sumEt (saved for backwards compatability (module name was HLTGlobalSums))
#	module hlt1MET350 = HLTGlobalSumMET {
#		InputTag inputTag = met
#		string observable = "sumEt"
#		double   Min = 350.0
#		double   Max = -1.0
#		int32    MinN = 1
#}
# SumEt filter
hlt1SumET120 = cms.EDFilter("HLTGlobalSumMET",
    observable = cms.string('sumEt'),
    Max = cms.double(-1.0),
    inputTag = cms.InputTag("met"),
    MinN = cms.int32(1),
    Min = cms.double(120.0)
)

# JET HT
hlt1HT350 = cms.EDFilter("HLTGlobalSumHT",
    observable = cms.string('sumEt'),
    Max = cms.double(-1.0),
    inputTag = cms.InputTag("htMet"),
    MinN = cms.int32(1),
    Min = cms.double(350.0)
)

hlt1HT400 = cms.EDFilter("HLTGlobalSumHT",
    observable = cms.string('sumEt'),
    Max = cms.double(-1.0),
    inputTag = cms.InputTag("htMet"),
    MinN = cms.int32(1),
    Min = cms.double(400.0)
)

# Double object filters (for acoplanarity cuts between 2 objects)
hlt2jetAco = cms.EDFilter("HLT2JetJet",
    MinMinv = cms.double(0.0),
    MinN = cms.int32(1),
    MaxMinv = cms.double(-1.0),
    MinDeta = cms.double(0.0),
    inputTag1 = cms.InputTag("hlt2jet125"),
    inputTag2 = cms.InputTag("hlt2jet125"),
    MaxDphi = cms.double(2.1),
    MaxDeta = cms.double(-1.0),
    MinDphi = cms.double(0.0)
)

hlt1jet1METAco = cms.EDFilter("HLT2JetMET",
    MinMinv = cms.double(0.0),
    MinN = cms.int32(1),
    MaxMinv = cms.double(-1.0),
    MinDeta = cms.double(0.0),
    inputTag1 = cms.InputTag("hlt1jet100"),
    inputTag2 = cms.InputTag("hlt1MET60"),
    MaxDphi = cms.double(2.1),
    MaxDeta = cms.double(-1.0),
    MinDphi = cms.double(0.0)
)

# HLT JetMET trigger sequences
JetMET1jet = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1jet")+pre1jet+cms.SequencePlaceholder("recoJetMETPath")+hlt1jet200)
JetMET1jetPE1 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1jetPE1")+pre1jetPE1+cms.SequencePlaceholder("recoJetMETPath")+hlt1jet150)
JetMET1jetPE3 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1jetPE3")+pre1jetPE3+cms.SequencePlaceholder("recoJetMETPath")+hlt1jet110)
JetMET1jetPE5 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1jetPE5")+pre1jetPE5+cms.SequencePlaceholder("recoJetMETPath")+hlt1jet60)
JetMET1jetPE7 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1jetPE7")+pre1jetPE7+cms.SequencePlaceholder("recoJetMETPath")+hlt1jet30)
JetMET2jet = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s2jet")+pre2jet+cms.SequencePlaceholder("recoJetMETPath")+hlt2jet150)
JetMET3jet = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s3jet")+pre3jet+cms.SequencePlaceholder("recoJetMETPath")+hlt3jet85)
JetMET4jet = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s4jet")+pre4jet+cms.SequencePlaceholder("recoJetMETPath")+hlt4jet60)
JetMET1MET = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1MET")+pre1MET+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET65)
JetMET1METPre1 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1METPre1")+pre1METPre1+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET55)
JetMET1METPre2 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1METPre2")+pre1METPre2+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET30)
JetMET1METPre3 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1METPre3")+pre1METPre3+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET20)
JetMET2jetAco = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s2jetAco")+pre2jetAco+cms.SequencePlaceholder("recoJetMETPath")+hlt2jet125+hlt2jetAco)
JetMET1jet1MET = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1jet1MET")+pre1jet1MET+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET60+hlt1jet180)
JetMET2jet1MET = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s2jet1MET")+pre2jet1MET+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET60+hlt2jet125)
JetMET3jet1MET = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s3jet1MET")+pre3jet1MET+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET60+hlt3jet60)
JetMET4jet1MET = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s4jet1MET")+pre4jet1MET+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET60+hlt4jet35)
JetMET1jet1METAco = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1jet1METAco")+pre1jet1METAco+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET60+hlt1jet100+hlt1jet1METAco)
JetMET2jetvbfMET = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s2jetvbfMET")+pre2jetvbfMET+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET60+hlt2jetvbf)
JetMETNV = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1snvMET")+prenv+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET60+hltnv)
JetMETPhi2METaco = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1sPhi2MET")+prephi2met+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET60+hltPhi2metAco)
JetMETPhiJet1METaco = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1sPhiJet1MET")+prephijet1met+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET70+hltPhiJet1metAco)
JetMETPhiJet2METaco = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1sPhiJet2MET")+prephijet2met+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET70+hltPhiJet2metAco)
JetMETPhiJet1Jet2aco = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1sPhiJet1Jet2")+prephijet1jet2+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET70+hltPhiJet1Jet2Aco)
JetMET1SumET = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1SumET")+pre1MET1SumET+cms.SequencePlaceholder("recoJetMETPath")+hlt1SumET120)
JetMET1HT = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1HT")+pre1MET1HT+cms.SequencePlaceholder("recoJetMETPath")+hlt1HT400)
JetMET1MET1HT = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1MET1HT")+pre1MET1HT+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET65+hlt1HT350)
JetMETRapGap = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1RapGap")+prerapgap+cms.SequencePlaceholder("recoJetMETPath")+hltRapGap)
JetMETDiJetAve30 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1sdijetave30")+predijetave30+cms.SequencePlaceholder("recoJetMETPath")+hltdijetave30)
JetMETDiJetAve60 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1sdijetave60")+predijetave60+cms.SequencePlaceholder("recoJetMETPath")+hltdijetave60)
JetMETDiJetAve110 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1sdijetave110")+predijetave110+cms.SequencePlaceholder("recoJetMETPath")+hltdijetave110)
JetMETDiJetAve150 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1sdijetave150")+predijetave150+cms.SequencePlaceholder("recoJetMETPath")+hltdijetave150)
JetMETDiJetAve200 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1sdijetave200")+predijetave200+cms.SequencePlaceholder("recoJetMETPath")+hltdijetave200)
hlt1jet30.MinPt = 30.0
hlt1jet50.MinPt = 50.0
hlt1jet60.MinPt = 60.0
hlt1jet100.MinPt = 100.0
hlt1jet110.MinPt = 110.0
hlt1jet120.MinPt = 120.0
hlt1jet150.MinPt = 150.0
hlt1jet180.MinPt = 180.0
hlt1jet200.MinPt = 200.0
hlt1jet250.MinPt = 250.0
hlt1jet400.MinPt = 400.0
hlt2jet125.MinPt = 125.0
hlt2jet125.MinN = 2
hlt2jet150.MinPt = 150.0
hlt2jet150.MinN = 2
hlt2jet155.MinPt = 155.0
hlt2jet155.MinN = 2
hlt2jet200.MinPt = 200.0
hlt2jet200.MinN = 2
hlt2jet350.MinPt = 350.0
hlt2jet350.MinN = 2
hlt2jetvbf.inputTag = 'MCJetCorJetIcone5'
hlt2jetvbf.minEt = 40.0
hlt2jetvbf.minDeltaEta = 2.5
hlt3jet60.MinPt = 60.0
hlt3jet60.MinN = 3
hlt3jet75.MinPt = 75.0
hlt3jet75.MinN = 3
hlt3jet85.MinPt = 85.0
hlt3jet85.MinN = 3
hlt3jet100.MinPt = 100.0
hlt3jet100.MinN = 3
hlt3jet195.MinPt = 195.0
hlt3jet195.MinN = 3
hlt4jet35.MinPt = 35.0
hlt4jet35.MinN = 4
hlt4jet50.MinPt = 50.0
hlt4jet50.MinN = 4
hlt4jet60.MinPt = 60.0
hlt4jet60.MinN = 4
hlt4jet80.MinPt = 80.0
hlt4jet80.MinN = 4
hlt1MET15.MinPt = 15.0
hlt1MET20.MinPt = 20.0
hlt1MET30.MinPt = 30.0
hlt1MET35.MinPt = 35.0
hlt1MET40.MinPt = 40.0
hlt1MET55.MinPt = 55.0
hlt1MET60.MinPt = 60.0
hlt1MET65.MinPt = 65.0
hlt1MET70.MinPt = 70.0
hlt1MET75.MinPt = 75.0
hlt1MET80.MinPt = 80.0
hlt1MET91.MinPt = 91.0
hltdijetave30.inputJetTag = 'MCJetCorJetIcone5'
hltdijetave30.minEtAve = 30.0
hltdijetave60.inputJetTag = 'MCJetCorJetIcone5'
hltdijetave60.minEtAve = 60.0
hltdijetave110.inputJetTag = 'MCJetCorJetIcone5'
hltdijetave110.minEtAve = 110.0
hltdijetave150.inputJetTag = 'MCJetCorJetIcone5'
hltdijetave150.minEtAve = 150.0
hltdijetave200.inputJetTag = 'MCJetCorJetIcone5'
hltdijetave200.minEtAve = 200.0
pre1METPre1.prescaleFactor = 100
pre1jetPE1.prescaleFactor = 10
predijetave150.prescaleFactor = 10

