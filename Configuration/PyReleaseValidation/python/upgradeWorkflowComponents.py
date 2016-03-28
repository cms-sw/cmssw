# You will need separate scenarios HERE for full and fast. DON'T CHANGE THE ORDER, only
# append new keys. Otherwise the numbering for the runTheMatrix tests will change.
upgradeKeys=['2017',
             '2017PU',
	     '2023',   
	     '2023dev', 
	     '2023sim',
	     '2023LReco',
	     '2023Reco' 
	     	       
	     
	     ]


upgradeGeoms={ '2017' : 'Extended2017',
	     '2023' : 'Extended2023',   
	     '2023dev' : 'Extended2023dev', 
	     '2023sim' : 'Extended2023sim',
	     '2023LReco': 'Extended2023LReco',
	     '2023Reco' : 'Extended2023Reco'
               }
	       
upgradeGTs={ '2017' : 'auto:phase1_2017_design',
	     '2023' :  'auto:run2_mc',
	     '2023dev' :  'auto:run2_mc',
	     '2023sim' : 'auto:run2_mc',
	     '2023LReco': 'auto:run2_mc',
	     '2023Reco' : 'auto:run2_mc'	     	      
             }
upgradeCustoms={ '2017' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017',
 		 '2023' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023',
 		 '2023dev' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023dev',
 		 '2023sim' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023sim',
 		 '2023LReco' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023LReco',
 		 '2023Reco' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023Reco'
		 
                 }
upgradeEras={ '2017' : 'Run2_2017',
	      '2023sim' : 'Run2_25ns',
	      '2023dev' : 'Run2_25ns'
              }

upgradeFragments=['FourMuPt_1_200_pythia8_cfi',
                  'SingleElectronPt10_cfi',
                  'SingleElectronPt35_cfi',
                  'SingleElectronPt1000_cfi',
                  'SingleGammaPt10_cfi',
                  'SingleGammaPt35_cfi',
                  'SingleMuPt1_cfi',
                  'SingleMuPt10_cfi',
                  'SingleMuPt100_cfi',
                  'SingleMuPt1000_cfi',
                  'FourMuExtendedPt_1_200_pythia8_cfi',
                  'TenMuExtendedE_0_200_pythia8_cfi',
                  'DoubleElectronPt10Extended_pythia8_cfi',
                  'DoubleElectronPt35Extended_pythia8_cfi',
                  'DoubleElectronPt1000Extended_pythia8_cfi',
                  'DoubleGammaPt10Extended_cfi',
                  'DoubleGammaPt35Extended_pythia8_cfi',
                  'DoubleMuPt1Extended_pythia8_cfi',
                  'DoubleMuPt10Extended_pythia8_cfi',
                  'DoubleMuPt100Extended_pythia8_cfi',
                  'DoubleMuPt1000Extended_pythia8_cfi',
                  'TenMuE_0_200_pythia8_cfi',
                  'SinglePiE50HCAL_cfi',
		  'MinBias_13TeV_pythia8_TuneCUETP8M1_cfi', 
		  'TTbar_13TeV_TuneCUETP8M1_cfi',
                  'ZEE_13TeV_TuneCUETP8M1_cfi',
                  'QCD_Pt_600_800_13TeV_TuneCUETP8M1_cfi',
                  'Wjet_Pt_80_120_14TeV_cfi',
                  'Wjet_Pt_3000_3500_14TeV_cfi',
                  'LM1_sfts_14TeV_cfi',
                  'QCD_Pt_3000_3500_14TeV_cfi',
                  'QCD_Pt_80_120_14TeV_cfi',
                  'H200ChargedTaus_Tauola_14TeV_cfi',
                  'JpsiMM_14TeV_cfi',
                  'TTbar_Tauola_14TeV_cfi',
                  'WE_14TeV_cfi',
                  'ZTT_Tauola_All_hadronic_14TeV_cfi',
                  'H130GGgluonfusion_14TeV_cfi',
                  'PhotonJet_Pt_10_14TeV_cfi',
                  'QQH1352T_Tauola_14TeV_cfi',
                  'MinBias_TuneZ2star_14TeV_pythia6_cff',
                  'WM_14TeV_cfi',
                  'ZMM_14TeV_cfi',
		  'QCDForPF_14TeV_cfi',
		  'DYToLL_M_50_TuneZ2star_14TeV_pythia6_tauola_cff',
		  'DYtoTauTau_M_50_TuneD6T_14TeV_pythia6_tauola_cff',
                 ]



### remember that you need to add a new step for phase 2 to include the track trigger
### remember that you need to add fastsim

# step1 is normal gen-sim
# step2 is digi
# step3 is reco
# step4 is harvest
# step5 is digi+l1tracktrigger
# step6 is fastsim
# step7 is fastsim harvesting
upgradeSteps=['GenSimFull','GenSimHLBeamSpotFull','DigiFull','RecoFull','RecoFullHGCAL','HARVESTFull','DigiTrkTrigFull','FastSim','HARVESTFast','DigiFullPU','RecoFullPU','RecoFullPUHGCAL','HARVESTFullPU','DigiFullTrigger']

upgradeScenToRun={ 
                   '2017':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   #'2017':['GenSimFull'],
		   '2017PU':['GenSimFull','DigiFullPU','RecoFullPU','HARVESTFullPU'],#full sequence
		   '2023':['GenSimFull','DigiFull','RecoFull'],#full sequence
		   '2023dev':['GenSimFull','DigiFull'],#dev scenario
		   '2023sim':['GenSimFull'],#sim scenario
		   '2023LReco':['GenSimFull','DigiFull'],#local reco scneario
		   '2023Reco':['GenSimFull','DigiFull','RecoFull']#full reco scenario
                   }

from  Configuration.PyReleaseValidation.relval_steps import Kby

howMuches={'FourMuPt_1_200_pythia8_cfi':Kby(10,100),
           'TenMuE_0_200_pythia8_cfi':Kby(10,100),
           'FourMuExtendedPt_1_200_pythia8_cfi':Kby(10,100),
           'TenMuExtendedE_0_200_pythia8_cfi':Kby(10,100),
           'SingleElectronPt10_cfi':Kby(9,100),
           'SingleElectronPt35_cfi':Kby(9,100),
           'SingleElectronPt1000_cfi':Kby(9,50),
           'SingleGammaPt10_cfi':Kby(9,100),
           'SingleGammaPt35_cfi':Kby(9,50),
           'SingleMuPt1_cfi':Kby(25,100),
           'SingleMuPt10_cfi':Kby(25,100),
           'SingleMuPt100_cfi':Kby(9,100),
           'SingleMuPt1000_cfi':Kby(9,100),
           'DoubleElectronPt10Extended_pythia8_cfi':Kby(9,100),
           'DoubleElectronPt35Extended_pythia8_cfi':Kby(9,100),
           'DoubleElectronPt1000Extended_pythia8_cfi':Kby(9,50),
           'DoubleGammaPt10Extended_cfi':Kby(9,100),
           'DoubleGammaPt35Extended_pythia8_cfi':Kby(9,50),
           'DoubleMuPt1Extended_pythia8_cfi':Kby(25,100),
           'DoubleMuPt10Extended_pythia8_cfi':Kby(25,100),
           'DoubleMuPt100Extended_pythia8_cfi':Kby(9,100),
           'DoubleMuPt1000Extended_pythia8_cfi':Kby(9,100),
           'SinglePiE50HCAL_cfi':Kby(10,100),
           'QCD_Pt_600_800_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'Wjet_Pt_80_120_14TeV_cfi':Kby(9,100),
           'Wjet_Pt_3000_3500_14TeV_cfi':Kby(9,50),
           'LM1_sfts_14TeV_cfi':Kby(9,100),
           'QCD_Pt_3000_3500_14TeV_cfi':Kby(9,50),
           'QCD_Pt_80_120_14TeV_cfi':Kby(9,100),
           'H200ChargedTaus_Tauola_14TeV_cfi':Kby(9,100),
           'JpsiMM_14TeV_cfi':Kby(66,100),
           'TTbar_Tauola_14TeV_cfi':Kby(9,100),
           'WE_14TeV_cfi':Kby(9,100),
           'ZEE_13TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'ZTT_Tauola_All_hadronic_14TeV_cfi':Kby(9,100),
           'H130GGgluonfusion_14TeV_cfi':Kby(9,100),
           'PhotonJet_Pt_10_14TeV_cfi':Kby(9,100),
           'QQH1352T_Tauola_14TeV_cfi':Kby(9,100),
           'MinBias_TuneZ2star_14TeV_pythia6_cff':Kby(90,100),
           'WM_14TeV_cfi':Kby(9,100),
           'ZMM_14TeV_cfi':Kby(18,100),
	   'QCDForPF_14TeV_cfi':Kby(9,50),
	   'DYToLL_M_50_TuneZ2star_14TeV_pythia6_tauola_cff':Kby(9,100),
	   'DYtoTauTau_M_50_TuneD6T_14TeV_pythia6_tauola_cff':Kby(9,100),
           'TTbar_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
	   'MinBias_13TeV_pythia8_TuneCUETP8M1_cfi':Kby(90,100)
           }

upgradeDatasetFromFragment={'FourMuPt_1_200_pythia8_cfi': 'FourMuPt1_200',
                            'FourMuExtendedPt_1_200_pythia8_cfi': 'FourMuExtendedPt1_200',
                            'TenMuE_0_200_pythia8_cfi': 'TenMuE_0_200',
                            'TenMuExtendedE_0_200_pythia8_cfi': 'TenMuExtendedE_0_200',
                            'SingleElectronPt10_cfi' : 'SingleElectronPt10',
                            'SingleElectronPt35_cfi' : 'SingleElectronPt35',
                            'SingleElectronPt1000_cfi' : 'SingleElectronPt1000',
                            'SingleGammaPt10_cfi' : 'SingleGammaPt10',
                            'SingleGammaPt35_cfi' : 'SingleGammaPt35',
                            'SingleMuPt1_cfi' : 'SingleMuPt1',
                            'SingleMuPt10_cfi' : 'SingleMuPt10',
                            'SingleMuPt100_cfi' : 'SingleMuPt100',
                            'SingleMuPt1000_cfi' : 'SingleMuPt1000',
                            'DoubleElectronPt10Extended_pythia8_cfi' : 'SingleElectronPt10Extended',
                            'DoubleElectronPt35Extended_pythia8_cfi' : 'SingleElectronPt35Extended',
                            'DoubleElectronPt1000Extended_pythia8_cfi' : 'SingleElectronPt1000Extended',
                            'DoubleGammaPt10Extended_cfi' : 'SingleGammaPt10Extended',
                            'DoubleGammaPt35Extended_pythia8_cfi' : 'SingleGammaPt35Extended',
                            'DoubleMuPt1Extended_pythia8_cfi' : 'SingleMuPt1Extended',
                            'DoubleMuPt10Extended_pythia8_cfi' : 'SingleMuPt10Extended',
                            'DoubleMuPt100Extended_pythia8_cfi' : 'SingleMuPt100Extended',
                            'DoubleMuPt1000Extended_pythia8_cfi' : 'SingleMuPt1000Extended',
                            'SinglePiE50HCAL_cfi' : 'SinglePiE50HCAL',
                            'QCD_Pt_600_800_13TeV_TuneCUETP8M1_cfi' : 'QCD_Pt_600_800_13',
                            'Wjet_Pt_80_120_14TeV_cfi' : 'Wjet_Pt_80_120_14TeV',
                            'Wjet_Pt_3000_3500_14TeV_cfi' : 'Wjet_Pt_3000_3500_14TeV',
                            'LM1_sfts_14TeV_cfi' : 'LM1_sfts_14TeV',
                            'QCD_Pt_3000_3500_14TeV_cfi' : 'QCD_Pt_3000_3500_14TeV',
                            'QCD_Pt_80_120_14TeV_cfi' : 'QCD_Pt_80_120_14TeV',
                            'H200ChargedTaus_Tauola_14TeV_cfi' : 'Higgs200ChargedTaus_14TeV',
                            'JpsiMM_14TeV_cfi' : 'JpsiMM_14TeV',
                            'TTbar_Tauola_14TeV_cfi' : 'TTbar_14TeV',
                            'WE_14TeV_cfi' : 'WE_14TeV',
                            'ZEE_13TeV_TuneCUETP8M1_cfi' : 'ZEE_13',
                            'ZTT_Tauola_All_hadronic_14TeV_cfi' : 'ZTT_14TeV',
                            'H130GGgluonfusion_14TeV_cfi' : 'H130GGgluonfusion_14TeV',
                            'PhotonJet_Pt_10_14TeV_cfi' : 'PhotonJets_Pt_10_14TeV',
                            'QQH1352T_Tauola_14TeV_cfi' : 'QQH1352T_Tauola_14TeV',
                            'MinBias_TuneZ2star_14TeV_pythia6_cff' : 'MinBias_TuneZ2star_14TeV',
                            'WM_14TeV_cfi' : 'WM_14TeV',
                            'ZMM_14TeV_cfi' : 'ZMM_14TeV',
			    'QCDForPF_14TeV_cfi' : 'QCDForPF_14TeV',
			    'DYToLL_M_50_TuneZ2star_14TeV_pythia6_tauola_cff' : 'DYToLL_M_50_TuneZ2star_14TeV',
			    'DYtoTauTau_M_50_TuneD6T_14TeV_pythia6_tauola_cff' : 'DYtoTauTau_M_50_TuneD6T_14TeV',
			    'TTbar_13TeV_TuneCUETP8M1_cfi' : 'TTbar_13',
			    'MinBias_13TeV_pythia8_TuneCUETP8M1_cfi' : 'MinBias_13'
                            }



#just do everything...
