import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
#JetMET filters to be used in JetMET HLT paths for luminosity L=2 x 10^33 cm-2s-1
# Single jet triggers
hlt1jet60 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt1jet100 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt1jet120 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt1jet180 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt1jet250 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt1jet400 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
# Double jet triggers
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
hlt3jet85 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt3jet195 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
# Quadruple jet triggers
hlt4jet35 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hlt4jet80 = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloMET_cfi import *
# MET
hlt1MET80 = copy.deepcopy(hlt1CaloMET)
import copy
from HLTrigger.HLTfilters.hlt1CaloMET_cfi import *
hlt1MET91 = copy.deepcopy(hlt1CaloMET)
import copy
from HLTrigger.JetMET.hltRapGapFilter_cfi import *
#Diffractive triggers
hltRapGap = copy.deepcopy(hltRapGapFilter)
import copy
from HLTrigger.JetMET.hltNVFilter_cfi import *
# New SUSY filters 
hltnv = copy.deepcopy(hltNVFilter)
import copy
from HLTrigger.JetMET.hltPhi2METFilter_cfi import *
hltPhi2metAco = copy.deepcopy(hltPhi2METFilter)
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
pre1MET1HT = copy.deepcopy(hltPrescaler)
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
prerapgap = copy.deepcopy(hltPrescaler)
# old MET HT using met sumEt (saved for backwards compatability (module name was HLTGlobalSums))
#	module hlt1MET350 = HLTGlobalSumMET {
#		InputTag inputTag = met
#		string observable = "sumEt"
#		double   Min = 350.0
#		double   Max = -1.0
#		int32    MinN = 1
#}
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
    inputTag1 = cms.InputTag("hlt2jet200"),
    inputTag2 = cms.InputTag("hlt2jet200"),
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
    inputTag2 = cms.InputTag("hlt1MET80"),
    MaxDphi = cms.double(2.1),
    MaxDeta = cms.double(-1.0),
    MinDphi = cms.double(0.0)
)

# HLT JetMET trigger sequences
JetMET1jet = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1jet")+pre1jet+cms.SequencePlaceholder("recoJetMETPath")+hlt1jet400)
JetMET2jet = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s2jet")+pre2jet+cms.SequencePlaceholder("recoJetMETPath")+hlt2jet350)
JetMET3jet = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s3jet")+pre3jet+cms.SequencePlaceholder("recoJetMETPath")+hlt3jet195)
JetMET4jet = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s4jet")+pre4jet+cms.SequencePlaceholder("recoJetMETPath")+hlt4jet80)
JetMET1MET = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1MET")+pre1MET+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET91)
JetMET2jetAco = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s2jetAco")+pre2jetAco+cms.SequencePlaceholder("recoJetMETPath")+hlt2jet200+hlt2jetAco)
JetMET1jet1METAco = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1jet1METAco")+pre1jet1METAco+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET80+hlt1jet100+hlt1jet1METAco)
JetMET1jet1MET = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1jet1MET")+pre1jet1MET+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET80+hlt1jet180)
JetMET2jet1MET = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s2jet1MET")+pre2jet1MET+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET80+hlt2jet155)
JetMET3jet1MET = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s3jet1MET")+pre3jet1MET+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET80+hlt3jet85)
JetMET4jet1MET = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s4jet1MET")+pre4jet1MET+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET80+hlt4jet35)
JetMET2jetvbfMET = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s2jetvbfMET")+pre2jetvbfMET+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET80+hlt2jetvbf)
JetMET1HT = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1HT")+pre1MET1HT+cms.SequencePlaceholder("recoJetMETPath")+hlt1HT400)
JetMET1MET1HT = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1MET1HT")+pre1MET1HT+cms.SequencePlaceholder("recoJetMETPath")+hlt1MET80+hlt1HT350)
JetMET1jetPE1 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1jetPE1")+pre1jetPE1+cms.SequencePlaceholder("recoJetMETPath")+hlt1jet250)
JetMET1jetPE3 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1jetPE3")+pre1jetPE3+cms.SequencePlaceholder("recoJetMETPath")+hlt1jet120)
JetMET1jetPE5 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1s1jetPE5")+pre1jetPE5+cms.SequencePlaceholder("recoJetMETPath")+hlt1jet60)
JetMETRapGap = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1RapGap")+prerapgap+cms.SequencePlaceholder("recoJetMETPath")+hltRapGap)
hlt1jet60.MinPt = 60.0
hlt1jet100.MinPt = 100.0
hlt1jet120.MinPt = 120.0
hlt1jet180.MinPt = 180.0
hlt1jet250.MinPt = 250.0
hlt1jet400.MinPt = 400.0
hlt2jet155.MinPt = 155.0
hlt2jet155.MinN = 2
hlt2jet200.MinPt = 200.0
hlt2jet200.MinN = 2
hlt2jet350.MinPt = 350.0
hlt2jet350.MinN = 2
hlt2jetvbf.inputTag = 'MCJetCorJetIcone5'
hlt3jet85.MinPt = 85.0
hlt3jet85.MinN = 3
hlt3jet195.MinPt = 195.0
hlt3jet195.MinN = 3
hlt4jet35.MinPt = 35.0
hlt4jet35.MinN = 4
hlt4jet80.MinPt = 80.0
hlt4jet80.MinN = 4
hlt1MET80.MinPt = 80.0
hlt1MET91.MinPt = 91.0
hltnv.inputMETTag = 'hlt1MET80'
hltnv.minEtJet1 = 80.
hltnv.minEtJet2 = 80.
hltPhi2METFilter.inputMETTag = 'hlt1MET80'
hltPhi2METFilter.minEtJet1 = 80.
hltPhi2METFilter.minEtJet2 = 80.

