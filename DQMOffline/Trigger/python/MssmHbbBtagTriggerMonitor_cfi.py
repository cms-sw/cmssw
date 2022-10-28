import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

triggerFlagPSet = cms.PSet(
  dcsInputTag = cms.InputTag('scalersRawToDigi'),
  dcsRecordInputTag = cms.InputTag('onlineMetaDataDigis'),
  dcsPartitions = cms.vint32( 24, 25, 26, 27, 28, 29 ),
  andOrDcs = cms.bool(False),
  errorReplyDcs = cms.bool(True),
  dbLabel = cms.string(''),
  andOrHlt = cms.bool(True),
  hltInputTag = cms.InputTag('TriggerResults', '', 'HLT'),
  hltPaths = cms.vstring(),
  hltDBKey = cms.string(''),
  errorReplyHlt = cms.bool(False),
  verbosityLevel = cms.uint32(1),
  andOr = cms.bool(False)
)

mssmHbbBtagTriggerMonitor = DQMEDAnalyzer("TagAndProbeBtagTriggerMonitor",
    dirname = cms.string("HLT/HIG/MssmHbb/"),
    requireValidHLTPaths = cms.bool(True),
    processname = cms.string("HLT"),
    jetPtMin = cms.double(40),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.7665),
    probeBtagMin = cms.double(0.4168),
    triggerobjbtag = cms.string("hltBTagCaloDeepCSV0p71Single8Jets30"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    btagAlgos = cms.VInputTag(cms.InputTag("pfDeepCSVJetTags:probb"), cms.InputTag("pfDeepCSVJetTags:probbb")),
    histoPSet = cms.PSet(
       jetPt  = cms.vdouble(40,45,50,55,60,65,70,75,80,85,90,95,100),
       jetEta = cms.vdouble(-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5),
       jetPhi = cms.vdouble(-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5),
       jetBtag = cms.vdouble(0.0,0.10,0.20,0.30,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.72,0.74,0.76,0.78,0.80,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.00),
    ),
    genericTriggerEventPSet = triggerFlagPSet.clone(),
)

# online btagging monitor

## Full hadronic - DeepCSV BTag

mssmHbbDeepCSVBtagTriggerMonitorFH40 = mssmHbbBtagTriggerMonitor.clone(
    dirname = "HLT/HIG/MssmHbb/control/btag/HLT_DoublePFJets40_PFBTagDeepCSV_p71",
    jetPtMin = 40,
    triggerobjbtag = "hltBTagPFDeepCSV0p71Single8Jets30",
    histoPSet = dict(jetPt = [40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,130,140,150,160,180,200,250,300,400,500]),
    genericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets40_PFBTagDeepCSV_p71_v*'])
)

mssmHbbDeepCSVBtagTriggerMonitorFH100 = mssmHbbBtagTriggerMonitor.clone(
    dirname = "HLT/HIG/MssmHbb/control/btag/HLT_DoublePFJets100_PFBTagDeepCSV_p71",
    jetPtMin = 100,
    triggerobjbtag = "hltBTagPFDeepCSV0p71Single8Jets30",
    histoPSet = dict(jetPt = [100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300,350,400,450,500,600,700,800]),
    genericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets100_PFBTagDeepCSV_p71_v*'])
)

mssmHbbDeepCSVBtagTriggerMonitorFH200 = mssmHbbBtagTriggerMonitor.clone(
    dirname = "HLT/HIG/MssmHbb/control/btag/HLT_DoublePFJets200_PFBTagDeepCSV_p71",
    jetPtMin = 200,
    triggerobjbtag = "hltBTagPFDeepCSV0p71Single8Jets30",
    histoPSet = dict(jetPt = [200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,380,400,420,440,460,500,550,600,700,800,900,1000]),
    genericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets200_PFBTagDeepCSV_p71_v*'])
)

mssmHbbDeepCSVBtagTriggerMonitorFH350 = mssmHbbBtagTriggerMonitor.clone(
    dirname = "HLT/HIG/MssmHbb/control/btag/HLT_DoublePFJets350_PFBTagDeepCSV_p71",
    jetPtMin = 350,
    triggerobjbtag = "hltBTagPFDeepCSV0p71Single8Jets30",
    histoPSet = dict(jetPt = [350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,650,700,750,800,900,1000]),
    genericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets350_PFBTagDeepCSV_p71_v*'])
)

## Semileptonic - DeepCSV BTag

mssmHbbDeepCSVBtagTriggerMonitorSL40 = mssmHbbBtagTriggerMonitor.clone(
    dirname = "HLT/HIG/MssmHbb/control/btag/HLT_Mu12_DoublePFJets40_PFBTagDeepCSV_p71",
    jetPtMin = 40,
    triggerobjbtag = "hltBTagPFDeepCSV0p71Single8Jets30",
    histoPSet = dict(jetPt = [40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,130,140,150,160,180,200,250,300,400,500]),
    genericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_DoublePFJets40_PFBTagDeepCSV_p71_v*'])
)

mssmHbbDeepCSVBtagTriggerMonitorSL100 = mssmHbbBtagTriggerMonitor.clone(
    dirname = "HLT/HIG/MssmHbb/control/btag/HLT_Mu12_DoublePFJets100_PFBTagDeepCSV_p71",
    jetPtMin = 100,
    triggerobjbtag = "hltBTagPFDeepCSV0p71Single8Jets30",
    histoPSet = dict(jetPt = [100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300,350,400,450,500,600,700,800]),
    genericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_DoublePFJets100_PFBTagDeepCSV_p71_v*'])
)

mssmHbbDeepCSVBtagTriggerMonitorSL200 = mssmHbbBtagTriggerMonitor.clone(
    dirname = "HLT/HIG/MssmHbb/control/btag/HLT_Mu12_DoublePFJets200_PFBTagDeepCSV_p71",
    jetPtMin = 200,
    triggerobjbtag = "hltBTagPFDeepCSV0p71Single8Jets30",
    histoPSet = dict(jetPt = [200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,380,400,420,440,460,500,550,600,700,800,900,1000]),
    genericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_DoublePFJets200_PFBTagDeepCSV_p71_v*'])
)

mssmHbbDeepCSVBtagTriggerMonitorSL350 = mssmHbbBtagTriggerMonitor.clone(
    dirname = "HLT/HIG/MssmHbb/control/btag/HLT_Mu12_DoublePFJets350_PFBTagDeepCSV_p71",
    jetPtMin = 350,
    triggerobjbtag = "hltBTagPFDeepCSV0p71Single8Jets30",
    histoPSet = dict(jetPt = [350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,650,700,750,800,900,1000]),
    genericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_DoublePFJets350_PFBTagDeepCSV_p71_v*'])
)


## Full hadronic - DeepJet BTag

mssmHbbDeepJetBtagTriggerMonitorFH40 = mssmHbbBtagTriggerMonitor.clone(
    dirname = "HLT/HIG/MssmHbb/control/btag/HLT_DoublePFJets40_PFBTagDeepJet_p71",
    jetPtMin = 40,
    triggerobjbtag = "hltBTagPFDeepJet0p71Single8Jets30",
    histoPSet = dict(jetPt = [40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,130,140,150,160,180,200,250,300,400,500]),
    genericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets40_PFBTagDeepJet_p71_v*'])
)

mssmHbbDeepJetBtagTriggerMonitorFH100 = mssmHbbBtagTriggerMonitor.clone(
    dirname = "HLT/HIG/MssmHbb/control/btag/HLT_DoublePFJets100_PFBTagDeepJet_p71",
    jetPtMin = 100,
    triggerobjbtag = "hltBTagPFDeepJet0p71Single8Jets30",
    histoPSet = dict(jetPt = [100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300,350,400,450,500,600,700,800]),
    genericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets100_PFBTagDeepJet_p71_v*'])
)

mssmHbbDeepJetBtagTriggerMonitorFH200 = mssmHbbBtagTriggerMonitor.clone(
    dirname = "HLT/HIG/MssmHbb/control/btag/HLT_DoublePFJets200_PFBTagDeepJet_p71",
    jetPtMin = 200,
    triggerobjbtag = "hltBTagPFDeepJet0p71Single8Jets30",
    histoPSet = dict(jetPt = [200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,380,400,420,440,460,500,550,600,700,800,900,1000]),
    genericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets200_PFBTagDeepJet_p71_v*'])
)

mssmHbbDeepJetBtagTriggerMonitorFH350 = mssmHbbBtagTriggerMonitor.clone(
    dirname = "HLT/HIG/MssmHbb/control/btag/HLT_DoublePFJets350_PFBTagDeepJet_p71",
    jetPtMin = 350,
    triggerobjbtag = "hltBTagPFDeepJet0p71Single8Jets30",
    histoPSet = dict(jetPt = [350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,650,700,750,800,900,1000]),
    genericTriggerEventPSet = dict(hltPaths = ['HLT_DoublePFJets350_PFBTagDeepJet_p71_v*'])
)


## Semileptonic - DeepJet BTag

mssmHbbDeepJetBtagTriggerMonitorSL40 = mssmHbbBtagTriggerMonitor.clone(
    dirname = "HLT/HIG/MssmHbb/control/btag/HLT_Mu12_DoublePFJets40_PFBTagDeepJet_p71",
    jetPtMin = 40,
    triggerobjbtag = "hltBTagPFDeepJet0p71Single8Jets30",
    histoPSet = dict(jetPt = [40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,130,140,150,160,180,200,250,300,400,500]),
    genericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_DoublePFJets40_PFBTagDeepJet_p71_v*'])
)

mssmHbbDeepJetBtagTriggerMonitorSL100 = mssmHbbBtagTriggerMonitor.clone(
    dirname = "HLT/HIG/MssmHbb/control/btag/HLT_Mu12_DoublePFJets100_PFBTagDeepJet_p71",
    jetPtMin = 100,
    triggerobjbtag = "hltBTagPFDeepJet0p71Single8Jets30",
    histoPSet = dict(jetPt = [100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300,350,400,450,500,600,700,800]),
    genericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_DoublePFJets100_PFBTagDeepJet_p71_v*'])
)

mssmHbbDeepJetBtagTriggerMonitorSL200 = mssmHbbBtagTriggerMonitor.clone(
    dirname = "HLT/HIG/MssmHbb/control/btag/HLT_Mu12_DoublePFJets200_PFBTagDeepJet_p71",
    jetPtMin = 200,
    triggerobjbtag = "hltBTagPFDeepJet0p71Single8Jets30",
    histoPSet = dict(jetPt = [200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,380,400,420,440,460,500,550,600,700,800,900,1000]),
    genericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_DoublePFJets200_PFBTagDeepJet_p71_v*'])
)

mssmHbbDeepJetBtagTriggerMonitorSL350 = mssmHbbBtagTriggerMonitor.clone(
    dirname = "HLT/HIG/MssmHbb/control/btag/HLT_Mu12_DoublePFJets350_PFBTagDeepJet_p71",
    jetPtMin = 350,
    triggerobjbtag = "hltBTagPFDeepJet0p71Single8Jets30",
    histoPSet = dict(jetPt = [350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,650,700,750,800,900,1000]),
    genericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_DoublePFJets350_PFBTagDeepJet_p71_v*'])
)

