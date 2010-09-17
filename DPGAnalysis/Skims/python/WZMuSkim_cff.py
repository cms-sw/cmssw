import FWCore.ParameterSet.Config as cms


# HLT filter
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
WZMuHLTFilter = copy.deepcopy(hltHighLevel)
WZMuHLTFilter.HLTPaths = ["HLT_Mu9","HLT_Mu11"]
WZMuHLTFilter.throw = False # don't throw on unknown path names


# Muon candidates filters 
qualityMuonFilter = cms.EDFilter("MuonSelector",
                                 src = cms.InputTag("muons"),
                                 cut = cms.string('pt > 20 && abs(eta)<2.4 && isGlobalMuon = 1 && isTrackerMuon = 1 && isolationR03().sumPt<3.0'),
                                 filter = cms.bool(True)                                
                                 )

# dxy filter on good muons
dxyMuonFilter = cms.EDFilter("MuonSelector",
                             src = cms.InputTag("goodMuons"),
                             cut = cms.string('abs(innerTrack().dxy)<1.0'),
                             filter = cms.bool(True)                                
                             )

# Z->mumu candidates
diMuonProducer = cms.EDProducer("CandViewShallowCloneCombiner",
                                checkCharge = cms.bool(True),
                                cut = cms.string('mass > 60'),
                                decay = cms.string("dxyMuonFilter@+ dxyMuonFilter@-")
                                )

# Z filters
diMuonFilter = cms.EDFilter("CandViewCountFilter",
                            src = cms.InputTag("dimuons"),
                            minNumber = cms.uint32(1)
                            )

# WMuNu candidates
from ElectroWeakAnalysis.WMuNu.wmunusProducer_cfi import *
# WMuNu candidates selectors
from ElectroWeakAnalysis.WMuNu.WMuNuSelection_cff import *
seltcMet.JetTag = cms.untracked.InputTag("ak5CaloJets")
seltcMet.TrigTag = cms.untracked.InputTag("TriggerResults::HLT")
seltcMet.IsCombinedIso = cms.untracked.bool(True)
seltcMet.IsoCut03 = cms.untracked.double(0.15)

selpfMet.JetTag = cms.untracked.InputTag("ak5CaloJets")
selpfMet.TrigTag = cms.untracked.InputTag("TriggerResults::HLT")
selpfMet.IsCombinedIso = cms.untracked.bool(True)
selpfMet.IsoCut03 = cms.untracked.double(0.15)


# define the sequences
diMuonSelSeq = cms.Sequence(WZMuHLTFilter *
                            qualityMuonFilter *
                            dxyMuonFilter *
                            diMuonProducer *
                            diMuonFilter
                            )

tcMetWMuNuSeq = cms.Sequence(WZMuHLTFilter *
                             tcMetWMuNus *
                             seltcMet
                             )

pfMetWMuNuSeq = cms.Sequence(WZMuHLTFilter *
                             pfMetWMuNus *
                             selpfMet
                             )


