import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

triggerFlagPSet = cms.PSet(
  dcsInputTag = cms.InputTag('scalersRawToDigi'),
  dcsPartitions = cms.vint32( 24, 25, 26, 27, 28, 29 ),
  andOrDcs = cms.bool(False),
  errorReplyDcs = cms.bool(True),
  dbLabel = cms.string(''),
  andOrHlt = cms.bool(True),
  hltInputTag = cms.InputTag('TriggerResults', '', 'HLT'),
  hltPaths = cms.vstring(),
  hltDBKey = cms.string(''),
  errorReplyHlt = cms.bool(False),
  verbosityLevel = cms.uint32(1)
)

mssmHbbBtagTriggerMonitor = DQMEDAnalyzer("TagAndProbeBtagTriggerMonitor",
    dirname = cms.string("HLT/HIG/MssmHbb/"),
    requireValidHLTPaths = cms.bool(True),
    processname = cms.string("HLT"),
    jetPtMin = cms.double(40),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.80),
    probeBtagMin = cms.double(0.45),
    triggerobjbtag = cms.string("hltBTagCaloDeepCSV0p71Single8Jets30"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    offlineBtag = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    histoPSet = cms.PSet(
       jetPt  = cms.vdouble(40,45,50,55,60,65,70,75,80,85,90,95,100),
       jetEta = cms.vdouble(-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5),
       jetPhi = cms.vdouble(-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5),
       jetBtag = cms.vdouble(0.80,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.00),
    ),
    genericTriggerEventPSet = triggerFlagPSet.clone(),
)

# online btagging monitor

mssmHbbBtagTriggerMonitorSL40noMu = mssmHbbBtagTriggerMonitor.clone()
mssmHbbBtagTriggerMonitorSL40noMu.dirname = cms.string("HLT/HIG/MssmHbb/semileptonic/BtagTrigger/pt40_noMuon")
mssmHbbBtagTriggerMonitorSL40noMu.jetPtMin = cms.double(40)
mssmHbbBtagTriggerMonitorSL40noMu.triggerobjbtag = cms.string("hltBTagCaloDeepCSV0p71Single8Jets30")
mssmHbbBtagTriggerMonitorSL40noMu.histoPSet.jetPt = cms.vdouble(40,45,50,55,60,65,70,75,80,85,90,95,100)
mssmHbbBtagTriggerMonitorSL40noMu.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoublePFJets40_CaloBTagDeepCSV_p71_v*')

mssmHbbBtagTriggerMonitorSL40 = mssmHbbBtagTriggerMonitor.clone()
mssmHbbBtagTriggerMonitorSL40.dirname = cms.string("HLT/HIG/MssmHbb/semileptonic/BtagTrigger/pt40")
mssmHbbBtagTriggerMonitorSL40.jetPtMin = cms.double(40)
mssmHbbBtagTriggerMonitorSL40.triggerobjbtag = cms.string("hltBTagCaloDeepCSV0p71Single8Jets30")
mssmHbbBtagTriggerMonitorSL40.histoPSet.jetPt = cms.vdouble(40,45,50,55,60,65,70,75,80,85,90,95,100)
mssmHbbBtagTriggerMonitorSL40.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu12_DoublePFJets40_CaloBTagDeepCSV_p71_v*')

mssmHbbBtagTriggerMonitorSL100 = mssmHbbBtagTriggerMonitor.clone()
mssmHbbBtagTriggerMonitorSL100.dirname = cms.string("HLT/HIG/MssmHbb/semileptonic/BtagTrigger/pt100")
mssmHbbBtagTriggerMonitorSL100.jetPtMin = cms.double(100)
mssmHbbBtagTriggerMonitorSL100.triggerobjbtag = cms.string("hltBTagCaloDeepCSV0p71Single8Jets30")
mssmHbbBtagTriggerMonitorSL100.histoPSet.jetPt = cms.vdouble(100,110,120,130,140,150,160,170,180,190,200)
mssmHbbBtagTriggerMonitorSL100.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu12_DoublePFJets100_CaloBTagDeepCSV_p71_v*')

mssmHbbBtagTriggerMonitorSL200 = mssmHbbBtagTriggerMonitor.clone()
mssmHbbBtagTriggerMonitorSL200.dirname = cms.string("HLT/HIG/MssmHbb/semileptonic/BtagTrigger/pt200")
mssmHbbBtagTriggerMonitorSL200.jetPtMin = cms.double(200)
mssmHbbBtagTriggerMonitorSL200.triggerobjbtag = cms.string("hltBTagCaloDeepCSV0p71Single8Jets30")
mssmHbbBtagTriggerMonitorSL200.histoPSet.jetPt = cms.vdouble(200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350)
mssmHbbBtagTriggerMonitorSL200.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu12_DoublePFJets200_CaloBTagDeepCSV_p71_v*')

mssmHbbBtagTriggerMonitorSL350 = mssmHbbBtagTriggerMonitor.clone()
mssmHbbBtagTriggerMonitorSL350.dirname = cms.string("HLT/HIG/MssmHbb/semileptonic/BtagTrigger/pt350")
mssmHbbBtagTriggerMonitorSL350.jetPtMin = cms.double(350)
mssmHbbBtagTriggerMonitorSL350.triggerobjbtag = cms.string("hltBTagCaloDeepCSV0p71Single8Jets30")
mssmHbbBtagTriggerMonitorSL350.histoPSet.jetPt = cms.vdouble(350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600)
mssmHbbBtagTriggerMonitorSL350.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu12_DoublePFJets350_CaloBTagDeepCSV_p71_v*')

mssmHbbBtagTriggerMonitorAH100 = mssmHbbBtagTriggerMonitor.clone()
mssmHbbBtagTriggerMonitorAH100.dirname = cms.string("HLT/HIG/MssmHbb/fullhadronic/BtagTrigger/pt100")
mssmHbbBtagTriggerMonitorAH100.jetPtMin = cms.double(100)
mssmHbbBtagTriggerMonitorAH100.triggerobjbtag = cms.string("hltBTagCaloDeepCSV0p71Single6Jets80")
mssmHbbBtagTriggerMonitorAH100.histoPSet.jetPt = cms.vdouble(100,110,120,130,140,150,160,170,180,190,200)
mssmHbbBtagTriggerMonitorAH100.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoublePFJets100_CaloBTagDeepCSV_p71_v*')

mssmHbbBtagTriggerMonitorAH200 = mssmHbbBtagTriggerMonitor.clone()
mssmHbbBtagTriggerMonitorAH200.dirname = cms.string("HLT/HIG/MssmHbb/fullhadronic/BtagTrigger/pt200")
mssmHbbBtagTriggerMonitorAH200.jetPtMin = cms.double(200)
mssmHbbBtagTriggerMonitorAH200.triggerobjbtag = cms.string("hltBTagCaloDeepCSV0p71Single6Jets80")
mssmHbbBtagTriggerMonitorAH200.histoPSet.jetPt = cms.vdouble(200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350)
mssmHbbBtagTriggerMonitorAH200.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoublePFJets200_CaloBTagDeepCSV_p71_v*')

mssmHbbBtagTriggerMonitorAH350 = mssmHbbBtagTriggerMonitor.clone()
mssmHbbBtagTriggerMonitorAH350.dirname = cms.string("HLT/HIG/MssmHbb/fullhadronic/BtagTrigger/pt350")
mssmHbbBtagTriggerMonitorAH350.jetPtMin = cms.double(350)
mssmHbbBtagTriggerMonitorAH350.triggerobjbtag = cms.string("hltBTagCaloDeepCSV0p71Single6Jets80")
mssmHbbBtagTriggerMonitorAH350.histoPSet.jetPt = cms.vdouble(350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600)
mssmHbbBtagTriggerMonitorAH350.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoublePFJets350_CaloBTagDeepCSV_p71_v*')
