import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("USER")
process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(-1)
)

process.load("ElectroWeakAnalysis.WMuNu.wmunusProducer_cfi")

process.source = cms.Source("PoolSource",
      fileNames = cms.untracked.vstring(
#      'file:/ciet3a/data1/cepeda/CMSSW_3_6_1_patch2/src/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks136100_1.root',
#      'file:/ciet3a/data1/cepeda/CMSSW_3_6_1_patch2/src/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks136100_2.root',
#      'file:/ciet3a/data1/cepeda/CMSSW_3_6_1_patch2/src/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks136098.root',
#      'file:/ciet3a/data1/cepeda/CMSSW_3_6_1_patch2/src/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks136097.root',
#      'file:/ciet3a/data1/cepeda/CMSSW_3_5_6/src/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks136080_1.root',
      'file:/ciet3b/data4/DataMu2010/SD_v9.root',
#      'file:/ciet3b/data4/DataMu2010/PD_v1.root', 
#      'file:/ciet3b/data4/DataMu2010/PD_v2.root',
     ),
      inputCommands = cms.untracked.vstring(
      'keep *', 'drop *_lumiProducer_*_*', 'drop *_MEtoEDMConverter_*_*', 'drop *_l1GtTriggerMenuLite_*_*' , 'drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap__HLT*'
      )
)
"""
import os,re
#file_directory ="/data4/Skimming/SkimResults/135/"
#file_directory="/data4/InclusiveMu15_Summer09-MC_31X_V3_AODSIM-v1/0024/"
#file_directory="/data1/degrutto/CMSSW_3_5_6/src/ElectroWeakAnalysis/Skimming/test/136"
#file_directory="/data1/degrutto/CMSSW_3_6_1_patch2/src/ElectroWeakAnalysis/Skimming/test/136/"
file_directory = "/ciet3b/data4/MuTemp/"

for file in os.listdir(file_directory):
     match = re.search(r'(root)', file)
     notmatch = re.search(r'(v2)', file)
     if notmatch: continue
     if not match: continue
     process.source.fileNames.append("file:" + file_directory + "/" + file)
"""
process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange(
      #'132440:157-132440:378',
      #'132596:382-132596:382',
      #'132596:447-132596:447',
      #'132598:174-132598:176',
      #'132599:1-132599:379',
      #'132599:381-132599:437',
      #'132601:1-132601:207',
      #'132601:209-132601:259',
      #'132601:261-132601:1107',
      #'132602:1-132602:70',
      #'132605:1-132605:444',
      #'132605:446-132605:522',
      #'132605:526-132605:622',
      #'132605:624-132605:814',
      #'132605:816-132605:829',
      #'132605:831-132605:867',
      #'132605:896-132605:942',
      #'132606:1-132606:26',
      #'132656:1-132656:111',
      #'132658:1-132658:51',
      #'132658:56-132658:120',
      #'132658:127-132658:148',
      #'132659:1-132659:76',
      #'132661:1-132661:116',
      #'132662:1-132662:9',
      #'132662:25-132662:74',
      #'132716:220-132716:436',
      #'132716:440-132716:487',
      #'132716:491-132716:586',
      #'132959:326-132959:334',
      #'132960:1-132960:124',
      #'132961:1-132961:222',
      #'132961:226-132961:230',
      #'132961:237-132961:381',
      #'132965:1-132965:68',
      #'132968:1-132968:67',
      #'132968:75-132968:169',
      #'133029:101-133029:115',
      #'133029:129-133029:332',
      #'133031:1-133031:18',
      #'133034:132-133034:287',
      #'133035:1-133035:63',
      #'133035:67-133035:302',
      #'133036:1-133036:222',
      #'133046:1-133046:43',
      #'133046:45-133046:210',
      #'133046:213-133046:227',
      #'133046:229-133046:323',
      #'133158:65-133158:786',
      '133874:166-133874:297',
      '133874:299-133874:721',
      '133874:724-133874:814',
      '133874:817-133874:864',
      '133875:1-133875:20',
      '133875:22-133875:37',
      '133876:1-133876:315',
      '133877:1-133877:77',
      '133877:82-133877:104',
      '133877:113-133877:231',
      '133877:236-133877:294',
      '133877:297-133877:437',
      '133877:439-133877:622',
      '133877:857-133877:1472',
      '133877:1474-133877:1640',
      '133877:1643-133877:1931',
      '133881:1-133881:71',
      '133881:74-133881:223',
      '133881:225-133881:551',
      '133885:1-133885:132',
      '133885:134-133885:728',
      '133927:1-133927:44',
      '133928:1-133928:645',
#      '135059:59-135059:67',
      '135149:297-135149:337',
      '135149:339-135149:754',
      '135149:756-135149:932',
      '135149:934-135149:937',
      '135149:942-135149:993',
      '135149:995-135149:1031',
      '135149:1033-135149:1098',
      '135149:1102-135149:1808',
      '135149:1811-135149:2269',
      '135149:2274-135149:2524',
      '135149:2528-135149:2713',
      '135149:2715-135149:3098',
      '135149:3100-135149:3102',
      '135149:3105-135149:3179',
      '135149:3182-135149:3303',
      '135149:3305-135149:3381',
      '135175:55-135175:545',
      '135175:548-135175:561',
      '135175:563-135175:790',
      '135175:792-135175:1042',
#      '135175:792-135175:1046',

      '135445:997-135445:1067',
      '135445:1069-135445:1329',
      '135445:1332-135445:1388',
      '135445:1391-135445:1629',
#      '135445:1631-135445:1827',
      '135445:1631-135445:1815',
      
      '135521:60-135521:108',
      '135521:110-135521:359',
      '135521:361-135521:440',
      '135521:442-135521:488',
      '135523:1-135523:64',
      '135523:66-135523:109',
      '135523:113-135523:124',
      '135523:126-135523:211',
      '135525:1-135525:3',
      '135525:6-135525:143',
      '135525:145-135525:381',
      '135525:384-135525:435',
      '135525:437-135525:452',
      '135528:1-135528:91',
      '135528:94-135528:95',
      '135528:98-135528:142',
      '135528:145-135528:147',
      '135528:149-135528:308',
      '135528:310-135528:454',
      '135528:456-135528:606',
      '135528:608-135528:609',
      '135528:611-135528:770',
      '135528:773-135528:776',
      '135528:779-135528:813',
      '135528:816-135528:912',
      '135528:915-135528:924',
      '135528:926-135528:1082',
      '135528:1084-135528:1213',
      '135528:1215-135528:1436',
      '135535:75-135535:167',
      '135535:169-135535:232',
      '135537:39-135537:69',
      '135573:102-135573:110',
      '135573:113-135573:118',
      '135573:120-135573:155',
      '135575:2-135575:210',
      '135575:213-135575:241',
      '135575:243-135575:264',
      '135575:266-135575:381',
      '135575:384-135575:638',
      '135575:645-135575:1161',
      '135575:1163-135575:1266',
#      '135575:1163-135575:1253',
      '135735:31-135735:42',
      '135735:44-135735:149',
      '135735:151-135735:234',
#      '135735:236-135735:320',

      #PD1
      '136033:138-136033:364','136033:366-136033:1102',
      '136035:1-136035:53',      '136035:55-136035:207',      '136035:209-136035:246',
      '136066:181-136066:297', '136066:299-136066:348', '136066:350-136066:529', '136066:532-136066:595','136066:597-136066:1177',
      '136080:1-136080:256',
      '136082:1-136082:506',


      #PD2
      '136087:1-136087:333',
      '136088:150-136088:256', 
      '136097:1-136097:91',
      '136098:1-136098:25',
      '136100:1-136100:1154',
      '136119:1-136119:36'
)



# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
      debugModules = cms.untracked.vstring('wmnVal_corMet','wmnVal_pfMet','wmnVal_tcMet','corMet'),
      cout = cms.untracked.PSet(
     #        default = cms.untracked.PSet( limit = cms.untracked.int32(20) ),
       #      threshold = cms.untracked.string('ERROR')
             threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)

process.goodMuons = cms.EDFilter("MuonSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('pt > 20.0 && isGlobalMuon=1'), # also || (isCaloMuon=1) ??
  filter = cms.bool(True)                                
) 

process.pt10Muons = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("corMetWMuNus"),
      MuonTrig=cms.untracked.string(""),
      PtThrForZ1 = cms.untracked.double(11110.0),
      PtThrForZ2 = cms.untracked.double(11110.0),
      EJetMin = cms.untracked.double(40.),
      NJetMax = cms.untracked.int32(999999),

      # Main cuts ->
      PtCut = cms.untracked.double(10.0),
      EtaCut = cms.untracked.double(4),
      IsRelativeIso = cms.untracked.bool(True),
      IsCombinedIso = cms.untracked.bool(True), #--> Changed default to Combined Iso. A cut in 0.15 is equivalent (for signal)
      IsoCut03 = cms.untracked.double(11111),    # to a cut in TrackIso in 0.10
      MtMin = cms.untracked.double(0.0),
      MtMax = cms.untracked.double(11100.0),
      MetMin = cms.untracked.double(-999999.),
      MetMax = cms.untracked.double(999999.),
      AcopCut = cms.untracked.double(5.),  # Remember to take this out if you are looking for High-Pt Bosons! (V+Jets)

      # Muon quality cuts ->
      DxyCut = cms.untracked.double(0.2),
      NormalizedChi2Cut = cms.untracked.double(10.),
      TrackerHitsCut = cms.untracked.int32(11),
      MuonHitsCut = cms.untracked.int32(1),
      IsAlsoTrackerMuon = cms.untracked.bool(True),

      # Select only W-, W+ ( default is all Ws)  
      SelectByCharge=cms.untracked.int32(0)
)





process.corMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("corMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      AcopCut = cms.untracked.double(2.0),
      PtCut = cms.untracked.double(20),
      MuonTrig=cms.untracked.string("HLT_Mu9")

)


process.pfMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("pfMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      AcopCut = cms.untracked.double(2.0),
      PtCut = cms.untracked.double(20),
      MuonTrig=cms.untracked.string("HLT_Mu9")

)


process.tcMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("tcMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      AcopCut = cms.untracked.double(2.0),
      PtCut = cms.untracked.double(20),
      MuonTrig=cms.untracked.string("HLT_Mu9")
)

process.dummy = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("tcMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      AcopCut = cms.untracked.double(2.0),
      PtCut = cms.untracked.double(20),
      MuonTrig=cms.untracked.string("HLT_Mu9"),
      MtMin = cms.untracked.double(0.0)
)



# Create a new reco::Muon collection with PFLow Iso information
process.muonsWithPFIso = cms.EDFilter("MuonWithPFIsoProducer",
        MuonTag = cms.untracked.InputTag("muons")
      , PfTag = cms.untracked.InputTag("particleFlow")
      , UsePfMuonsOnly = cms.untracked.bool(False)
      , TrackIsoVeto = cms.untracked.double(0.01)
      , GammaIsoVeto = cms.untracked.double(0.07)
      , NeutralHadronIsoVeto = cms.untracked.double(0.1)
)

process.pfWMuNus = cms.EDProducer("WMuNuProducer",
      # Input collections ->
      MuonTag = cms.untracked.InputTag("muonsWithPFIso"),
      METTag = cms.untracked.InputTag("pfMet")
)


process.pfMetAndMuons = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      MuonTag = cms.untracked.InputTag("muonsWithPFIso"),
      WMuNuCollectionTag = cms.untracked.InputTag("pfWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      AcopCut = cms.untracked.double(2.0),
      PtCut = cms.untracked.double(15),
      MuonTrig=cms.untracked.string("HLT_Mu9")
)

process.load("ElectroWeakAnalysis.WMuNu.wmunusValidation_cfi") #load validation sequence (for WMunu & ZMuMu)
process.wmnVal_pfMet.JetTag = cms.untracked.InputTag("ak5CaloJets")
process.wmnVal_corMet.JetTag = cms.untracked.InputTag("ak5CaloJets") 
process.wmnVal_tcMet.JetTag = cms.untracked.InputTag("ak5CaloJets") 
process.wmnVal_tcMet.PtCut =  cms.untracked.double(20.)
process.wmnVal_pfMet.PtCut =  cms.untracked.double(20.)
process.wmnVal_corMet.PtCut =  cms.untracked.double(20.)
process.wmnVal_tcMet.MuonTrig =  cms.untracked.string("HLT_Mu9")
process.wmnVal_pfMet.MuonTrig =  cms.untracked.string("HLT_Mu9")
process.wmnVal_tcMet.MuonTrig =  cms.untracked.string("HLT_Mu9")
process.wmnVal_pfMet.MuonTrig =  cms.untracked.string("HLT_Mu9")
process.wmnVal_corMet.MuonTrig =  cms.untracked.string("HLT_Mu9")

process.wmnVal_rawcaloMet=process.wmnVal_corMet.clone()
process.wmnVal_rawcaloMet.METTag = cms.untracked.InputTag("met")
process.wmnVal_rawcaloMet.METIncludesMuons= cms.untracked.bool(False)

process.wmnVal_pfMetAndMuons=process.wmnVal_pfMet.clone()
process.wmnVal_pfMetAndMuons.JetTag = cms.untracked.InputTag("ak5CaloJets")
process.wmnVal_pfMetAndMuons.MuonTag=cms.untracked.InputTag("muonsWithPFIso")


#process.load("UserCode.GPetrucc.edmLumi_cfi")

process.TFileService = cms.Service("TFileService", fileName = cms.string('runs_SD_pt20_HLTMu9.root') )

process.eventDump = cms.EDAnalyzer(
    "EventDumper",
 #    srcMuons = cms.InputTag("goodMuonsPt15")
    )

# Steering the process
process.path1 = cms.Path(process.wmnVal_corMet)
process.path2 = cms.Path(process.wmnVal_pfMet)
process.path3 = cms.Path(process.wmnVal_tcMet)
process.path4 = cms.Path(process.wmnVal_rawcaloMet)

process.path5 = cms.Path(process.corMetWMuNus+process.corMet)
process.path6 = cms.Path(process.pfMetWMuNus+process.pfMet)
process.path7 = cms.Path(process.tcMetWMuNus+process.tcMet)






