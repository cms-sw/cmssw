import FWCore.ParameterSet.Config as cms

import copy
from HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi import *
# HLT Btag Softmuon paths analyers
hltb1jetmu = copy.deepcopy(hltBtagLifetimeAnalyzer)
import copy
from HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi import *
hltb2jetmu = copy.deepcopy(hltBtagLifetimeAnalyzer)
import copy
from HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi import *
hltb3jetmu = copy.deepcopy(hltBtagLifetimeAnalyzer)
import copy
from HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi import *
hltb4jetmu = copy.deepcopy(hltBtagLifetimeAnalyzer)
import copy
from HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi import *
hltbhtmu = copy.deepcopy(hltBtagLifetimeAnalyzer)
hltBSoftmuonModules = cms.PSet(
    modules = cms.vstring('hltb1jetmu', 
        'hltb2jetmu', 
        'hltb3jetmu', 
        'hltb4jetmu', 
        'hltbhtmu')
)
hltBSoftmuon = cms.Sequence(hltb1jetmu+hltb2jetmu+hltb3jetmu+hltb4jetmu+hltbhtmu)
hltb1jetmu.triggerPath = 'HLTB1Jet'
hltb1jetmu.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBSoftmuonNjetL1seeds"),
    jets = cms.InputTag("iterativeCone5CaloJets","","HLT"),
    name = cms.string('preL2'),
    title = cms.string('pre-L2')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuon1jetL2filter"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL25filter"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonByDRL3filter"),
        jets = cms.InputTag("hltBSoftmuonHLTJetsByDR"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hltb2jetmu.triggerPath = 'HLTB2Jet'
hltb2jetmu.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBSoftmuonNjetL1seeds"),
    jets = cms.InputTag("iterativeCone5CaloJets","","HLT"),
    name = cms.string('preL2'),
    title = cms.string('pre-L2')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuon2jetL2filter"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL25filter"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL3filter"),
        jets = cms.InputTag("hltBSoftmuonHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hltb3jetmu.triggerPath = 'HLTB3Jet'
hltb3jetmu.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBSoftmuonNjetL1seeds"),
    jets = cms.InputTag("iterativeCone5CaloJets","","HLT"),
    name = cms.string('preL2'),
    title = cms.string('pre-L2')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuon3jetL2filter"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL25filter"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL3filter"),
        jets = cms.InputTag("hltBSoftmuonHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hltb4jetmu.triggerPath = 'HLTB4Jet'
hltb4jetmu.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBSoftmuonNjetL1seeds"),
    jets = cms.InputTag("iterativeCone5CaloJets","","HLT"),
    name = cms.string('preL2'),
    title = cms.string('pre-L2')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuon4jetL2filter"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL25filter"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL3filter"),
        jets = cms.InputTag("hltBSoftmuonHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hltbhtmu.triggerPath = 'HLTBHT'
hltbhtmu.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBSoftmuonHTL1seeds"),
    jets = cms.InputTag("iterativeCone5CaloJets","","HLT"),
    name = cms.string('preL2'),
    title = cms.string('pre-L2')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonHTL2filter"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL25filter"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL3filter"),
        jets = cms.InputTag("hltBSoftmuonHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))

