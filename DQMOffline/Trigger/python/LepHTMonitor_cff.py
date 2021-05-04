import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from copy import deepcopy


### Single Electron + HT triggers
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
DQMOffline_Ele15_HT600 = DQMEDAnalyzer('LepHTMonitor',
                                              electronCollection = cms.InputTag('gedGsfElectrons'),
                                              electronVID = cms.InputTag("egmGsfElectronIDsForDQM:cutBasedElectronID-Summer16-80X-V1-medium"),
                                              muonCollection = cms.InputTag(''),
                                              muonIDlevel = cms.untracked.int32(2), ## 1: loose, 2: medium, 3: tight
                                              pfMetCollection = cms.InputTag('pfMet'),
                                              pfJetCollection = cms.InputTag('ak4PFJets'),
                                              jetTagCollection = cms.InputTag(''),

                                              vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                              conversionCollection = cms.InputTag('conversions'),
                                              beamSpot = cms.InputTag('offlineBeamSpot'),

                                              folderName = cms.string('HLT_Ele15_IsoVVVL_PFHT600'),
                                              requireValidHLTPaths = cms.bool(True),

                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(2.5),
                                              metCut = cms.untracked.double(-1.0),
                                              htCut = cms.untracked.double(-1.0), 
                                              nels = cms.untracked.double(1),
                                              nmus = cms.untracked.double(0),
                                              leptonPtPlateau = cms.untracked.double(30.0), #defines plateau region for eta, phi distributions
                                              leptonCountingThreshold = cms.untracked.double(10.0), # min lepton pT for lepton counting
                                              lepIsoCut = cms.untracked.double(0.1), # max lepton reliso 
                                              lepEtaCut = cms.untracked.double(2.5), # max abs(eta)
                                              lep_d0_cut_b = cms.untracked.double(0.0118), #barrel
                                              lep_dz_cut_b = cms.untracked.double(0.373),
                                              lep_d0_cut_e = cms.untracked.double(0.0739), #endcap
                                              lep_dz_cut_e = cms.untracked.double(0.602), 
                                                                                           
                                              ptbins = cms.vdouble(0,5,10,20,30,40,50,75,100,125,160,200,250),
                                              htbins = cms.vdouble(0,50,100,150,200,250,300,350,400,450,500,600,750,1000,1500,2000),
                                              nbins_eta = cms.untracked.int32(10),
                                              nbins_phi = cms.untracked.int32(10),
                                              nbins_npv = cms.untracked.int32(35),
                                              etabins_min = cms.untracked.double(-2.5), 
                                              etabins_max = cms.untracked.double(2.5), 
                                              phibins_min = cms.untracked.double(-3.142), 
                                              phibins_max = cms.untracked.double(3.142), 
                                              npvbins_min = cms.untracked.double(0), 
                                              npvbins_max = cms.untracked.double(70), 
                                                
                                              numGenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_Ele15_IsoVVVL_PFHT600_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                verbosityLevel = cms.uint32(0)
                                                ),
                                              den_lep_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_PFHT1050_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(0)
                                                ),
                                              den_HT_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_Ele38_WPTight_Gsf_v*","HLT_Ele27_WPTight_Gsf_v*","HLT_Ele35_WPTight_Gsf_v*","HLT_Ele40_WPTight_Gsf_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(0)
                                                ),
                                            )

DQMOffline_Ele15_HT450 = DQMOffline_Ele15_HT600.clone()
DQMOffline_Ele15_HT450.folderName =  cms.string('HLT_Ele15_IsoVVVL_PFHT450')
DQMOffline_Ele15_HT450.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Ele15_IsoVVVL_PFHT450_v*")

DQMOffline_Ele50_HT450 = DQMOffline_Ele15_HT600.clone()
DQMOffline_Ele50_HT450.folderName =  cms.string('HLT_Ele50_IsoVVVL_PFH450')
DQMOffline_Ele50_HT450.leptonPtPlateau = cms.untracked.double(60.0)
DQMOffline_Ele50_HT450.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Ele50_IsoVVVL_PFHT450_v*")

### Single Muon + HT triggers
DQMOffline_Mu15_HT600 = DQMOffline_Ele15_HT600.clone()
DQMOffline_Mu15_HT600.electronCollection = cms.InputTag('')
DQMOffline_Mu15_HT600.conversionCollection = cms.InputTag('')
DQMOffline_Mu15_HT600.muonCollection = cms.InputTag('muons')
DQMOffline_Mu15_HT600.muonIDlevel = cms.untracked.int32(2) ## 1: loose, 2: medium, 3: tight
DQMOffline_Mu15_HT600.nels = cms.untracked.double(0)
DQMOffline_Mu15_HT600.nmus = cms.untracked.double(1)
DQMOffline_Mu15_HT600.lepIsoCut = cms.untracked.double(0.2) 
DQMOffline_Mu15_HT600.lepEtaCut = cms.untracked.double(2.4)
DQMOffline_Mu15_HT600.lep_d0_cut_b = cms.untracked.double(0.2) #endcap parameter not used for muons 
DQMOffline_Mu15_HT600.lep_dz_cut_b = cms.untracked.double(0.5)
                                              
DQMOffline_Mu15_HT600.folderName =  cms.string('HLT_Mu15_IsoVVVL_PFH600')
DQMOffline_Mu15_HT600.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu15_IsoVVVL_PFHT600_v*")
DQMOffline_Mu15_HT600.den_HT_GenericTriggerEventPSet.hltPaths = cms.vstring("HLT_IsoMu27_v*","HLT_IsoMu24_v*")

DQMOffline_Mu15_HT450 = DQMOffline_Mu15_HT600.clone()
DQMOffline_Mu15_HT450.folderName =  cms.string('HLT_Mu15_IsoVVVL_PFHT450')
DQMOffline_Mu15_HT450.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu15_IsoVVVL_PFHT450_v*")

DQMOffline_Mu50_HT450 = DQMOffline_Mu15_HT600.clone()
DQMOffline_Mu50_HT450.folderName =  cms.string('HLT_Mu50_IsoVVVL_PFH450')
DQMOffline_Mu50_HT450.leptonPtPlateau = cms.untracked.double(60.0)
DQMOffline_Mu50_HT450.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu50_IsoVVVL_PFHT450_v*")

### Dilepton + HT triggers
DQMOffline_DoubleMu4_Mass8_DZ_PFHT350 = DQMOffline_Mu15_HT600.clone()
DQMOffline_DoubleMu4_Mass8_DZ_PFHT350.nmus = cms.untracked.double(2)
DQMOffline_DoubleMu4_Mass8_DZ_PFHT350.folderName =  cms.string('HLT_DoubleMu4_Mass8_DZ_PFHT350')
DQMOffline_DoubleMu4_Mass8_DZ_PFHT350.leptonPtPlateau = cms.untracked.double(6.0)
DQMOffline_DoubleMu4_Mass8_DZ_PFHT350.leptonCountingThreshold = cms.untracked.double(4.0)
DQMOffline_DoubleMu4_Mass8_DZ_PFHT350.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_Mass8_DZ_PFHT350_v*")
DQMOffline_DoubleMu4_Mass8_DZ_PFHT350.den_lep_GenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu15_IsoVVVL_PFHT450_v*")
DQMOffline_DoubleMu4_Mass8_DZ_PFHT350.den_HT_GenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_v*")

DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350 = DQMOffline_Ele15_HT600.clone()
DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350.nels = cms.untracked.double(2)
DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350.folderName =  cms.string('HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350')
DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350.leptonPtPlateau = cms.untracked.double(10.0)
DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350.leptonCountingThreshold = cms.untracked.double(8.0)
DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350_v*")
DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350.den_lep_GenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Ele15_IsoVVVL_PFHT450_v*")
DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350.den_HT_GenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ*")

DQMOffline_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ = DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350.clone()
DQMOffline_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ.muonCollection = cms.InputTag('muons')
DQMOffline_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ.nels = cms.untracked.double(1)
DQMOffline_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ.nmus = cms.untracked.double(1)
DQMOffline_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ.folderName =  cms.string('HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ')
DQMOffline_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ_v*")
DQMOffline_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ.den_lep_GenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Ele15_IsoVVVL_PFHT450_v*","HLT_Mu15_IsoVVVL_PFHT450_v*")
DQMOffline_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ.den_HT_GenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*", "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v*","HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*")

### Alternate dilepton + HT without DZ requirement
DQMOffline_DoubleMu4_Mass8_PFHT350 = DQMOffline_DoubleMu4_Mass8_DZ_PFHT350.clone()
DQMOffline_DoubleMu4_Mass8_PFHT350.folderName =  cms.string('HLT_DoubleMu4_Mass8_PFHT350')
DQMOffline_DoubleMu4_Mass8_PFHT350.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_Mass8_PFHT350_v*")

DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350 = DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350.clone()
DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350.folderName =  cms.string('HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350')
DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350_v*")

DQMOffline_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350 = DQMOffline_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ.clone()
DQMOffline_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350.folderName =  cms.string('HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350')
DQMOffline_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_v*")

# fastsim has no conversion collection
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(DQMOffline_Ele15_HT600,conversionCollection=cms.InputTag(''))
fastSim.toModify(DQMOffline_Ele15_HT450,conversionCollection=cms.InputTag(''))
fastSim.toModify(DQMOffline_Ele50_HT450,conversionCollection=cms.InputTag(''))
fastSim.toModify(DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350,conversionCollection=cms.InputTag(''))
fastSim.toModify(DQMOffline_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ,conversionCollection=cms.InputTag(''))

DQMOffline_LepHT_POSTPROCESSING = DQMEDHarvester("DQMGenericClient",
                                                             subDirs = cms.untracked.vstring('HLT/SUSY/LepHT/*'),
                                                             efficiency = cms.vstring(
        "lepPtTurnOn_eff ';Offline lepton p_{T} [GeV];#epsilon' lepPtTurnOn_num lepPtTurnOn_den",
        "lepEtaTurnOn_eff ';Offline lepton #eta;#epsilon' lepEtaTurnOn_num lepEtaTurnOn_den",
        "lepPhiTurnOn_eff ';Offline lepton #phi;#epsilon' lepPhiTurnOn_num lepPhiTurnOn_den",
        "lepEtaPhiTurnOn_eff ';Offline lepton #eta;Offline lepton #phi;#epsilon' lepEtaPhiTurnOn_num lepEtaPhiTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den",
        "NPVTurnOn_eff ';Offline N_{PV} ;#epsilon' NPVTurnOn_num NPVTurnOn_den"
        ),
                                                             resolution = cms.vstring('')
                                                             )



from DQMOffline.Trigger.HLTEGTnPMonitor_cfi import egmGsfElectronIDsForDQM

LepHTMonitor = cms.Sequence( 
                              DQMOffline_Ele15_HT600
                            + DQMOffline_Ele15_HT450
                            + DQMOffline_Ele50_HT450
                            + DQMOffline_Mu15_HT600
                            + DQMOffline_Mu15_HT450
                            + DQMOffline_Mu50_HT450
                            + DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350
                            + DQMOffline_DoubleMu4_Mass8_DZ_PFHT350
                            + DQMOffline_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ
                            + DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350
                            + DQMOffline_DoubleMu4_Mass8_PFHT350
                            + DQMOffline_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350
                            , cms.Task(egmGsfElectronIDsForDQM) # Use of electron VID requires this module being executed first
)

LepHTClient = cms.Sequence(  DQMOffline_LepHT_POSTPROCESSING )
