#!/bin/sh

cmsDriver.py Pyquen_DiJet_pt80to120_2760GeV_cfi --conditions auto:run2_mc_HIon -s GEN:pgen_himix,SIM --datatier GEN-SIM -n 1 --relval 2000,1 --eventcontent RAWSIM --scenario HeavyIons --beamspot MatchHI --pileup HiMixGEN --pileup_input das:/RelValQCD_Pt_80_120_13/CMSSW_7_2_0_pre7-PRE_LS172_V11-v1/GEN-SIM --fileout "step1_GEN_SIM_PU.root"

cmsDriver.py step2  --conditions auto:run2_mc_HIon -s DIGI:pdigi_valid,L1,DIGI2RAW,RAW2DIGI,L1Reco --scenario HeavyIons --datatier GEN-SIM-DIGI-RAW-HLTDEBUG --pileup HiMix --pileup_input das:/RelValQCD_Pt_80_120_13/CMSSW_7_2_0_pre7-PRE_LS172_V11-v1/GEN-SIM  -n 20 --eventcontent FEVTDEBUGHLT --filein "file:step1_GEN_SIM_PU.root" --fileout "step2_DIGI_L1_DIGI2RAW_HLT_RAW2DIGI_L1Reco_PU.root"

cmsDriver.py step3  --conditions auto:run2_mc_HIon --scenario HeavyIons -n 20 --eventcontent RECOSIM -s RAW2DIGI,L1Reco,RECO --datatier GEN-SIM-RECO --pileup HiMix --pileup_input das:/RelValQCD_Pt_80_120_13/CMSSW_7_2_0_pre7-PRE_LS172_V11-v1/GEN-SIM --filein "file:step2_DIGI_L1_DIGI2RAW_HLT_RAW2DIGI_L1Reco_PU.root"

cmsRun validateHiMixing.py > mix.out 2> mix.err

