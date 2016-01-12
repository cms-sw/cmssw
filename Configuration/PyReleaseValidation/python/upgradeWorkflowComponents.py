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
	     '2023sim' : 'Extended2023',
	     '2023LReco': 'Extended2023',
	     '2023Reco' : 'Extended2023'
               }
	       
upgradeGTs={ '2017' : 'auto:phase1_2017_design',
	     '2023' :  'auto_run2_mc',
	     '2023dev' :  'auto_run2_mc',
	     '2023sim' : 'auto_run2_mc',
	     '2023LReco': 'auto_run2_mc',
	     '2023Reco' : 'auto_run2_mc'	     	      
             }
upgradeCustoms={ '2017' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017',
 		 '2023' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023',
 		 '2023dev' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023dev',
 		 '2023sim' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023sim',
 		 '2023LReco' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023LReco',
 		 '2023Reco' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023Reco'
		 
                 }

upgradeFragments=['FourMuPt_1_200_pythia8_cfi','SingleElectronPt10_cfi',
                  'SingleElectronPt35_cfi','SingleElectronPt1000_cfi',
                  'SingleGammaPt10_cfi','SingleGammaPt35_cfi','SingleMuPt1_cfi','SingleMuPt10_cfi',
                  'SingleMuPt100_cfi','SingleMuPt1000_cfi','TTbarLepton_Tauola_8TeV_cfi','Wjet_Pt_80_120_8TeV_cfi',
                  'Wjet_Pt_3000_3500_8TeV_cfi','LM1_sfts_8TeV_cfi','QCD_Pt_3000_3500_8TeV_cfi','QCD_Pt_600_800_8TeV_cfi',
                  'QCD_Pt_80_120_8TeV_cfi','H200ChargedTaus_Tauola_8TeV_cfi','JpsiMM_8TeV_cfi','TTbar_Tauola_8TeV_cfi',
                  'WE_8TeV_cfi','ZEE_8TeV_cfi','ZTT_Tauola_All_hadronic_8TeV_cfi','H130GGgluonfusion_8TeV_cfi',
                  'PhotonJet_Pt_10_8TeV_cfi','QQH1352T_Tauola_8TeV_cfi','MinBias_TuneZ2star_8TeV_pythia6_cff','WM_8TeV_cfi',
                  'ZMM_8TeV_cfi','ADDMonoJet_8TeV_d3MD3_cfi','ZpMM_8TeV_cfi','WpM_8TeV_cfi',
                  'Wjet_Pt_80_120_14TeV_cfi','Wjet_Pt_3000_3500_14TeV_cfi','LM1_sfts_14TeV_cfi','QCD_Pt_3000_3500_14TeV_cfi',
                  'QCD_Pt_80_120_14TeV_cfi','H200ChargedTaus_Tauola_14TeV_cfi','JpsiMM_14TeV_cfi','TTbar_Tauola_14TeV_cfi',
                  'WE_14TeV_cfi','ZEE_14TeV_cfi','ZTT_Tauola_All_hadronic_14TeV_cfi','H130GGgluonfusion_14TeV_cfi',
                  'PhotonJet_Pt_10_14TeV_cfi','QQH1352T_Tauola_14TeV_cfi',
                  'MinBias_TuneZ2star_14TeV_pythia6_cff','WM_14TeV_cfi','ZMM_14TeV_cfi',
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
		  'QCDForPF_14TeV_cfi',
		  'DYToLL_M_50_TuneZ2star_14TeV_pythia6_tauola_cff',
		  'DYtoTauTau_M_50_TuneD6T_14TeV_pythia6_tauola_cff',
		  'TTbar_14TeV_TuneCUETP8M1_cfi',
		  'MinBias_14TeV_pythia8_TuneCUETP8M1_cfi'		  ]



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

upgradeScenToRun={ '2017':['GenSimFull','DigiFull','RecoFull'],#HARVESTING REMOVED
		   '2023':['GenSimFull','DigiFull','RecoFull'],#full sequence
		   '2023dev':['GenSimFull','DigiFull','RecoFull'],#dev scenario
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
           'TTbarLepton_Tauola_8TeV_cfi':Kby(9,100),
           'Wjet_Pt_80_120_8TeV_cfi':Kby(9,100),
           'Wjet_Pt_3000_3500_8TeV_cfi':Kby(9,50),
           'LM1_sfts_8TeV_cfi':Kby(9,100),
           'QCD_Pt_3000_3500_8TeV_cfi':Kby(9,50),
           'QCD_Pt_600_800_8TeV_cfi':Kby(9,50),
           'QCD_Pt_80_120_8TeV_cfi':Kby(9,100),
           'H200ChargedTaus_Tauola_8TeV_cfi':Kby(9,100),
           'JpsiMM_8TeV_cfi':Kby(66,100),
           'TTbar_Tauola_8TeV_cfi':Kby(9,100),
           'WE_8TeV_cfi':Kby(9,100),
           'ZEE_8TeV_cfi':Kby(9,100),
           'ZTT_Tauola_All_hadronic_8TeV_cfi':Kby(9,100),
           'H130GGgluonfusion_8TeV_cfi':Kby(9,100),
           'PhotonJet_Pt_10_8TeV_cfi':Kby(9,100),
           'QQH1352T_Tauola_8TeV_cfi':Kby(9,100),
           'MinBias_TuneZ2star_8TeV_pythia6_cff':Kby(9,30),
           'WM_8TeV_cfi':Kby(9,100),
           'ZMM_8TeV_cfi':Kby(18,100),
           'ADDMonoJet_8TeV_d3MD3_cfi':Kby(9,100),
           'ZpMM_8TeV_cfi':Kby(9,100),
           'WpM_8TeV_cfi':Kby(9,100),
           'Wjet_Pt_80_120_14TeV_cfi':Kby(9,100),
           'Wjet_Pt_3000_3500_14TeV_cfi':Kby(9,50),
           'LM1_sfts_14TeV_cfi':Kby(9,100),
           'QCD_Pt_3000_3500_14TeV_cfi':Kby(9,50),
           'QCD_Pt_80_120_14TeV_cfi':Kby(9,100),
           'H200ChargedTaus_Tauola_14TeV_cfi':Kby(9,100),
           'JpsiMM_14TeV_cfi':Kby(66,100),
           'TTbar_Tauola_14TeV_cfi':Kby(9,100),
           'WE_14TeV_cfi':Kby(9,100),
           'ZEE_14TeV_cfi':Kby(9,100),
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
           'TTbar_14TeV_TuneCUETP8M1_cfi':Kby(9,50),
	   'MinBias_14TeV_pythia8_TuneCUETP8M1_cfi':Kby(90,100)
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
                            'TTbarLepton_Tauola_8TeV_cfi' : 'TTbarLepton_8TeV',
                            'Wjet_Pt_80_120_8TeV_cfi' : 'Wjet_Pt_80_120_8TeV',
                            'Wjet_Pt_3000_3500_8TeV_cfi' : 'Wjet_Pt_3000_3500_8TeV',
                            'LM1_sfts_8TeV_cfi' : 'LM1_sfts_8TeV',
                            'QCD_Pt_3000_3500_8TeV_cfi' : 'QCD_Pt_3000_3500_8TeV',
                            'QCD_Pt_600_800_8TeV_cfi' : 'QCD_Pt_600_800_8TeV',
                            'QCD_Pt_80_120_8TeV_cfi' : 'QCD_Pt_80_120_8TeV',
                            'H200ChargedTaus_Tauola_8TeV_cfi' : 'Higgs200ChargedTaus_8TeV',
                            'JpsiMM_8TeV_cfi' : 'JpsiMM_8TeV',
                            'TTbar_Tauola_8TeV_cfi' : 'TTbar_8TeV',
                            'WE_8TeV_cfi' : 'WE_8TeV',
                            'ZEE_8TeV_cfi' : 'ZEE_8TeV',
                            'ZTT_Tauola_All_hadronic_8TeV_cfi' : 'ZTT_8TeV',
                            'H130GGgluonfusion_8TeV_cfi' : 'H130GGgluonfusion_8TeV',
                            'PhotonJet_Pt_10_8TeV_cfi' : 'PhotonJets_Pt_10_8TeV',
                            'QQH1352T_Tauola_8TeV_cfi' : 'QQH1352T_Tauola_8TeV',
                            'MinBias_TuneZ2star_8TeV_pythia6_cff': 'MinBias_TuneZ2star_8TeV',
                            'WM_8TeV_cfi' : 'WM_8TeV',
                            'ZMM_8TeV_cfi' : 'ZMM_8TeV',
                            'ADDMonoJet_8TeV_d3MD3_cfi' : 'ADDMonoJet_d3MD3_8TeV',
                            'ZpMM_8TeV_cfi' : 'ZpMM_8TeV',
                            'WpM_8TeV_cfi' : 'WpM_8TeV',
                            'Wjet_Pt_80_120_14TeV_cfi' : 'Wjet_Pt_80_120_14TeV',
                            'Wjet_Pt_3000_3500_14TeV_cfi' : 'Wjet_Pt_3000_3500_14TeV',
                            'LM1_sfts_14TeV_cfi' : 'LM1_sfts_14TeV',
                            'QCD_Pt_3000_3500_14TeV_cfi' : 'QCD_Pt_3000_3500_14TeV',
                            'QCD_Pt_80_120_14TeV_cfi' : 'QCD_Pt_80_120_14TeV',
                            'H200ChargedTaus_Tauola_14TeV_cfi' : 'Higgs200ChargedTaus_14TeV',
                            'JpsiMM_14TeV_cfi' : 'JpsiMM_14TeV',
                            'TTbar_Tauola_14TeV_cfi' : 'TTbar_14TeV',
                            'WE_14TeV_cfi' : 'WE_14TeV',
                            'ZEE_14TeV_cfi' : 'ZEE_14TeV',
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
			    'TTbar_14TeV_TuneCUETP8M1_cfi' : 'TTbar_pythia8_14TeV',
			    'MinBias_14TeV_pythia8_TuneCUETP8M1_cfi' : 'MinBias_pythia8_14TeV'
                            }



#just do everything...
