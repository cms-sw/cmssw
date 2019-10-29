#! /bin/tcsh

cmsenv

rehash

#
# old files in castor: rfdir /castor/cern.ch/cms/store/...
# new files in eos   : cmsLs /store/...
# new files in eos   : eos ls /store/...
#

#
# gen sim input files for Monte-Carlo tests
#   InputGenSimGRun0 = /store/relval/CMSSW_8_0_11/RelValProdTTbar/GEN-SIM/80X_mcRun1_realistic_v4-v1/10000/06A6C86B-C634-E611-93A5-0CC47A74525A.root
set InputGenSimGRun0 = root://eoscms.cern.ch//eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/STORM/GEN-SIM/CMSSW_8/06A6C86B-C634-E611-93A5-0CC47A74525A.root
#   InputGenSimGRun1 = /store/relval/CMSSW_8_0_16/RelValProdTTbar_13/GEN-SIM/80X_mcRun2_asymptotic_v16_gs7120p2-v1/10000/06F2C3AC-8957-E611-9DDF-0025905B85D8.root
set InputGenSimGRun1 = root://eoscms.cern.ch//eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/STORM/GEN-SIM/CMSSW_8/06F2C3AC-8957-E611-9DDF-0025905B85D8.root
#   InputGenSimGRun2 = /store/relval/CMSSW_8_0_16/RelValProdTTbar_13/GEN-SIM/80X_mcRun2_asymptotic_v16_gs7120p2-v1/10000/06F2C3AC-8957-E611-9DDF-0025905B85D8.root
set InputGenSimGRun2 = root://eoscms.cern.ch//eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/STORM/GEN-SIM/CMSSW_8/06F2C3AC-8957-E611-9DDF-0025905B85D8.root
#   InputGenSimGRun3 = /store/relval/CMSSW_9_0_0_pre5/RelValTTbar_13/GEN-SIM/90X_upgrade2017_realistic_v15-v1/00000/14F749AC-8AFE-E611-9821-0CC47A78A4A0.root
set InputGenSimGRun3 = root://eoscms.cern.ch//eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/STORM/GEN-SIM/CMSSW_9_phase1/14F749AC-8AFE-E611-9821-0CC47A78A4A0.root
#   InputGenSimHIon1 = /store/relval/CMSSW_8_0_16/RelValZEEMM_13_HI/GEN-SIM/80X_mcRun2_HeavyIon_v9-v1/10000/F8FC5F64-1657-E611-A57E-002590A887F0.root
set InputGenSimHIon1 = root://eoscms.cern.ch//eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/STORM/GEN-SIM/CMSSW_8/F8FC5F64-1657-E611-A57E-002590A887F0.root
#   InputGenSimHIon3 = /store/relval/CMSSW_10_3_0_pre5/RelValZEEMM_13_HI/GEN-SIM/103X_upgrade2018_realistic_v7-v1/10000/E288668E-A2D1-D446-A401-D71EA43DD796.root
set InputGenSimHIon3 = root://eoscms.cern.ch//eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/STORM/GEN-SIM/CMSSW_10/E288668E-A2D1-D446-A401-D71EA43DD796.root
set InputGenSimPIon2 = $InputGenSimGRun2
set InputGenSimPRef2 = $InputGenSimGRun2
set InputGenSimPIon3 = $InputGenSimGRun3
set InputGenSimPRef3 = $InputGenSimGRun3
#
# lhc raw input files for Real-Data tests
set InputLHCRawGRun0 = root://eoscms.cern.ch//eos/cms/store/data/Run2012A/MuEG/RAW/v1/000/191/718/14932935-E289-E111-830C-5404A6388697.root
set InputLHCRawGRun1 = root://eoscms.cern.ch//eos/cms/store/data/Run2015D/MuonEG/RAW/v1/000/256/677/00000/80950A90-745D-E511-92FD-02163E011C5D.root
set InputLHCRawGRun2 = root://eoscms.cern.ch//eos/cms/store/data/Run2016B/JetHT/RAW/v1/000/272/762/00000/C666CDE2-E013-E611-B15A-02163E011DBE.root
set InputLHCRawGRun3 = root://eoscms.cern.ch//eos/cms/store/data/Run2017A/HLTPhysics4/RAW/v1/000/295/606/00000/36DE5E0A-3645-E711-8FA1-02163E01A43B.root
set InputLHCRawHIon1 = root://eoscms.cern.ch//eos/cms/store/hidata/HIRun2015/HIHardProbes/RAW-RECO/HighPtJet-PromptReco-v1/000/263/689/00000/1802CD9A-DDB8-E511-9CF9-02163E0138CA.root
#et InputLHCRawHIon3 = root://eoscms.cern.ch//eos/cms/store/hidata/HIRun2018A/HIHardProbes/RAW/v1/000/326/479/00000/853DBE29-53BA-9A44-9FDD-58E4E9064EB1.root
set InputLHCRawHIon3 = root://eoscms.cern.ch//eos/cms/store/data/Run2018D/HIMinimumBias0/RAW/v1/000/325/112/00000/660F62BB-9932-D645-A4A4-0BBBDA3963E8.root
set InputLHCRawPIon2 = $InputLHCRawGRun2
set InputLHCRawPRef2 = $InputLHCRawGRun2
set InputLHCRawPIon3 = $InputLHCRawGRun3
set InputLHCRawPRef3 = $InputLHCRawGRun3

#
# global tags to be used
set BASE1MC  = auto:run1_mc
set BASE1HLT = auto:run1_hlt
set BASE1RD  = auto:run1_data
set BASE2MC  = auto:run2_mc
set BASE2HLT = auto:run2_hlt
set BASE2RD  = auto:run2_data

set NNPPMC = 100
set NNPPRD = 100
set NNHIMC = 100
set NNHIRD = 25

set EraRun1        = " "
set EraRun25ns     = " --era=Run2_25ns "
set EraRun2pp2016  = " --era=Run2_2016 "
set EraRun2pp2017  = " --era=Run2_2017 "
set EraRun2pp2018  = " --era=Run2_2018 "
set EraRun2HImc    = " --era=Run2_2018_pp_on_AA "
set EraRun2HIrd    = " --era=Run2_2018_pp_on_AA "
 
set XL1T    = "" # syntax: tag,record[,connect,label]
set XL1TPP1 = "" # "L1GtTriggerMenu_L1Menu_Collisions2012_v1_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_CONDITIONS"
set XL1TPP2 = "" # "L1GtTriggerMenu_L1Menu_Collisions2012_v2_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_CONDITIONS"
set XL1TPP3 = "" # "L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_CONDITIONS"
#set XL1TPP3 = "L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc,L1GtTriggerMenuRcd,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2012_v3/sqlFile/L1Menu_Collisions2012_v3_mc.db"
set XL1THI  = "" # "L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2011_v0_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_CONDITIONS"
#set XL1THI = "L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2011_v0_mc,L1GtTriggerMenuRcd,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_CollisionsHeavyIons2011_v0/sqlFile/L1Menu_CollisionsHeavyIons2011_v0_mc.db"
set XL1TPI  = "" # "L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2013_v0_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_CONDITIONS"
#set XL1TPI =  "L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2013_v0_mc,L1GtTriggerMenuRcd,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_CollisionsHeavyIons2013_v0/sqlFile/L1Menu_CollisionsHeavyIons2013_v0_mc.db" 
set XL1TLOWPU  = "" # ""

# specific workflows, first varying the globaltags, then the hlt tables

# Append new JECs (as long as not in GT):
#set XJEC = "JetCorrectorParametersCollection_HLT_V1_AK4Calo,JetCorrectionsRecord,frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS,AK4CaloHLT+JetCorrectorParametersCollection_HLT_trk1B_V1_AK4PF,JetCorrectionsRecord,frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS,AK4PFHLT"
#set XJEC = "JetCorrectorParametersCollection_AK5Calo_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_CONDITIONS,AK5CaloHLT+JetCorrectorParametersCollection_AK5PF_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_CONDITIONS,AK5PFHLT+JetCorrectorParametersCollection_AK5PFchs_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_CONDITIONS,AK5PFchsHLT"
#set XJEC = "JetCorrectorParametersCollection_AK5PF_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_CONDITIONS,AK5PFHLT+JetCorrectorParametersCollection_AK5PFchs_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_CONDITIONS,AK5PFchsHLT"

#set XL1TPP1 = ${XL1TPP1}+${XJEC}
#set XL1TPP2 = ${XL1TPP2}+${XJEC}
#set XL1TPP3 = ${XJEC}
#set XL1THI  = ${XJEC}
#set XL1TPI  = ${XJEC}

foreach gtag ( MC DATA )
  if ( $gtag == DATA ) then
    set BASE1  = $BASE1HLT
    set BASE2  = $BASE2HLT
    set NNPP   = $NNPPRD
    set NNHI   = $NNHIRD
    set DATAMC = --data
    set PNAME  = HLT1
    set RNAME  = RECO1
    set EraRun2HI = $EraRun2HIrd
  else  if ( $gtag == MC ) then
    set BASE1  = $BASE1MC
    set BASE2  = $BASE2MC
    set NNPP   = $NNPPMC
    set NNHI   = $NNHIMC
    set DATAMC = --mc
    set PNAME  = HLT
    set RNAME  = RECO
    set EraRun2HI = $EraRun2HImc
  else
    # unsupported
    continue
  endif
  
  if ( $1 == "" ) then
    set tables = ( GRun )
  else if ( ($1 == all) || ($1 == ALL) ) then
    set tables = ( GRun HIon PIon PRef Fake Fake1 Fake2 2018 )
  else if ( ($1 == ib) || ($1 == IB) ) then
    set tables = ( GRun HIon PIon PRef )
  else if ( ($1 == dev) || ($1 == DEV) ) then
    set tables = ( GRun HIon PIon PRef )
  else if ( ($1 == full) || ($1 == FULL) ) then
    set tables = ( FULL )
  else if ( ($1 == fake) || ($1 == FAKE) ) then
    set tables = ( Fake Fake1 Fake2 )
  else if ( ($1 == frozen) || ($1 == FROZEN) ) then
    set tables = ( Fake Fake1 Fake2 2018 )
  else
    set tables = ( $1 )
  endif
  
  foreach table ( $tables )

    set name = ${table}_${gtag}  

    if ( $table == FULL ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:FULL
      set GTAG = ${BASE2}_FULL
      set RTAG = ${BASE2RD}_FULL
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun3
      set InputLHCRaw = $InputLHCRawGRun3
      set Era  = $EraRun2pp2018
      set Custom = " "
      set L1REPACK = L1REPACK:Full
      set DIGI = DIGI:pdigi
    else if ( $table == Fake ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:Fake
      set GTAG = ${BASE1}_Fake
      set RTAG = ${BASE1RD}_Fake
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun0
      set InputLHCRaw = $InputLHCRawGRun0
      set Era  = $EraRun1
      set Custom = " "
      set L1REPACK = L1REPACK:GT1
      set DIGI = DIGI:pdigi
    else if ( $table == Fake1 ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:Fake1
      set GTAG = ${BASE2}_Fake1
      set RTAG = ${BASE2RD}_Fake1
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun1
      set InputLHCRaw = $InputLHCRawGRun1
      set Era  = $EraRun25ns
      set Custom = " "
      set L1REPACK = L1REPACK:GCTGT
      set DIGI = DIGI:pdigi
    else if ( $table == Fake2 ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:Fake2
      set GTAG = ${BASE2}_Fake2
      set RTAG = ${BASE2RD}_Fake2
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun2
      set InputLHCRaw = $InputLHCRawGRun2
      set Era  = $EraRun2pp2016
      set Custom = " "
      set L1REPACK = L1REPACK:Full
      set DIGI = DIGI:pdigi
    else if ( $table == 2018 ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:2018
      set GTAG = ${BASE2}_2018
      set RTAG = ${BASE2RD}_2018
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun3
      set InputLHCRaw = $InputLHCRawGRun3
      set Era  = $EraRun2pp2018
      set Custom = " "
      set L1REPACK = L1REPACK:Full
      set DIGI = DIGI:pdigi
    else if ( $table == GRun ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:GRun
      set GTAG = ${BASE2}_GRun
      set RTAG = ${BASE2RD}_GRun
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun3
      set InputLHCRaw = $InputLHCRawGRun3
      set Era  = $EraRun2pp2018
      set Custom = " "
      set L1REPACK = L1REPACK:Full
      set DIGI = DIGI:pdigi
    else if ( $table == HIon ) then
      set XL1T = $XL1THI
      set XHLT = HLT:HIon
      set GTAG = ${BASE2}_HIon
      set RTAG = ${BASE2RD}_HIon
      set NN   = $NNHI
      set SCEN = pp
      set InputGenSim = $InputGenSimHIon3
      set InputLHCRaw = $InputLHCRawHIon3
      set Era  = $EraRun2HI
      set Custom = " "
      set L1REPACK = L1REPACK:Full
      set DIGI = DIGI:pdigi_hi
    else if ( $table == PIon ) then
      set XL1T = $XL1TPI
      set XHLT = HLT:PIon
      set GTAG = ${BASE2}_PIon
      set RTAG = ${BASE2RD}_PIon
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimPIon3
      set InputLHCRaw = $InputLHCRawPIon3
      set Era  = $EraRun2pp2018
      set Custom = " "
      set L1REPACK = L1REPACK:Full
      set DIGI = DIGI:pdigi
    else if ( $table == PRef ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:PRef
      set GTAG = ${BASE2}_PRef
      set RTAG = ${BASE2RD}_PRef
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimPRef3
      set InputLHCRaw = $InputLHCRawPRef3
      set Era  = $EraRun2pp2018
      set Custom = " "
      set L1REPACK = L1REPACK:Full
      set DIGI = DIGI:pdigi
    else
      # unsupported
      continue
    endif


    if ( $gtag == DATA ) then

    echo
    echo "Creating L1RePack $name"

    if ( $table == HIon ) then
    cmsDriver.py RelVal                 --step=$L1REPACK                                   --conditions=$GTAG --filein=$InputLHCRaw                        --custom_conditions=$XL1T --fileout=RelVal_L1RePack_$name.root      --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW'               --eventcontent=RAW                     --customise=HLTrigger/Configuration/CustomConfigs.L1T     $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_L1RePack_$name.py --customise=L1Trigger/Configuration/L1Trigger_custom.customiseResetPrescalesAndMasks --customise_commands='from FWCore.ParameterSet.MassReplace import massSearchReplaceAnyInputTag; massSearchReplaceAnyInputTag(process.SimL1Emulator,"rawDataCollector","rawDataRepacker",False,True)'
    else
    cmsDriver.py RelVal                 --step=$L1REPACK                                   --conditions=$GTAG --filein=$InputLHCRaw                        --custom_conditions=$XL1T --fileout=RelVal_L1RePack_$name.root      --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW'               --eventcontent=RAW                     --customise=HLTrigger/Configuration/CustomConfigs.L1T     $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_L1RePack_$name.py --customise=L1Trigger/Configuration/L1Trigger_custom.customiseResetPrescalesAndMasks
    endif

    else

#   echo
#   echo "Creating TTbarGenToHLT $name"
#   cmsDriver.py TTbar_Tauola_13TeV_TuneCUETP8M1_cfi --step=GEN,SIM,DIGI,L1,DIGI2RAW,$XHLT              --conditions=$GTAG                                              --custom_conditions=$XL1T  --fileout=RelVal_GenSim_$name.root       --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW-HLT'           --eventcontent=FEVTDEBUGHLT            --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_GenSim_$name.py

    echo
    echo "Creating DigiL1Raw $name"
    cmsDriver.py RelVal                 --step=$DIGI,L1,DIGI2RAW                           --conditions=$GTAG --filein=$InputGenSim                        --custom_conditions=$XL1T  --fileout=RelVal_DigiL1Raw_$name.root    --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW'               --eventcontent=RAWSIM                  --customise=HLTrigger/Configuration/CustomConfigs.L1T     $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_DigiL1Raw_$name.py

    cat >>RelVal_DigiL1Raw_$name.py<<EOF
process.options.numberOfThreads = cms.untracked.uint32(1)
process.options.numberOfStreams = cms.untracked.uint32(1)
EOF

    echo
    echo "Creating DigiL1RawHLT $name"
    cmsDriver.py RelVal                 --step=$DIGI,L1,DIGI2RAW,$XHLT                     --conditions=$GTAG --filein=$InputGenSim                        --custom_conditions=$XL1T  --fileout=RelVal_DigiL1RawHLT_$name.root --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW-HLT'           --eventcontent=FEVTDEBUGHLT            --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_DigiL1RawHLT_$name.py  --processName=$PNAME

    cat >>RelVal_DigiL1RawHLT_$name.py<<EOF
process.options.numberOfThreads = cms.untracked.uint32(1)
process.options.numberOfStreams = cms.untracked.uint32(1)
EOF

#   echo
#   echo "Creating FastSim $name"
#   cmsDriver.py TTbar_Tauola_13TeV_TuneCUETP8M1_cfi --step=GEN,SIM,RECOBEFMIX,DIGI,L1,DIGI2RAW,L1Reco,RECO,$XHLT --fast --conditions=$GTAG                             --custom_conditions=$XL1T  --fileout=FastSim_GenToHLT_$name.root    --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RECO'              --eventcontent FEVTDEBUGHLT            --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  $Era --customise=$Custom  --scenario=$SCEN --python_filename=FastSim_GenToHLT_$name.py     --processName=$PNAME

    endif

    echo
    echo "Creating HLT $name"
    cmsDriver.py RelVal                 --step=$XHLT                                       --conditions=$GTAG --filein=file:RelVal_Raw_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_HLT_$name.root          --number=$NN $DATAMC --no_exec --datatier 'SIM-DIGI-RAW-HLTDEBUG'          --eventcontent=FEVTDEBUGHLT            --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_HLT_$name.py           --processName=$PNAME

    echo
    echo "Creating HLT2 (re-running HLT) $name"
    cmsDriver.py RelVal                 --step=$XHLT                                       --conditions=$GTAG --filein=file:RelVal_HLT_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_HLT2_$name.root         --number=$NN $DATAMC --no_exec --datatier 'SIM-DIGI-RAW-HLTDEBUG'          --eventcontent=RAW                     --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_HLT2_$name.py          --processName=HLT2

    if ( $gtag == DATA ) then

    echo
    echo "Creating HLT+L1Reco+RECO $name"
    cmsDriver.py RelVal                 --step=$XHLT,RAW2DIGI,L1Reco,RECO                  --conditions=$RTAG --filein=file:RelVal_Raw_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_HLT_RECO_$name.root     --number=$NN $DATAMC --no_exec --datatier 'SIM-RAW-HLT-RECO'               --eventcontent=RAW                     --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_HLT_Reco_$name.py      --processName=$PNAME

    if ( $table == HIon ) then
      set STEPS = "RAW2DIGI,L1Reco,RECO,EI,DQM"
    else
      set STEPS = "RAW2DIGI,L1Reco,RECO,EI,PAT,DQM"
    endif
    echo
    echo "Creating RECO+EI+PAT+DQM $name"
    cmsDriver.py RelVal                 --step=$STEPS                                      --conditions=$RTAG --filein=file:RelVal_HLT_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_RECO_$name.root         --number=$NN $DATAMC --no_exec --datatier 'RECO,MINIAOD,DQMIO'             --eventcontent=RECO,MINIAOD,DQM        --customise=HLTrigger/Configuration/CustomConfigs.Base    $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_RECO_$name.py          --processName=$RNAME  --runUnscheduled

    else

    set RTAG = $GTAG

    if ( $table == HIon ) then
      set STEPS = "RAW2DIGI,L1Reco,RECO,EI,PAT,VALIDATION,DQM"
      set CustomCommand = ""
    else
      set STEPS = "RAW2DIGI,L1Reco,RECO,EI,PAT,VALIDATION,DQM"
      set CustomCommand = "--customise_commands=process.CSCHaloData.HLTResultLabel=cms.InputTag('')"
    endif

    echo
    echo "Creating HLT+L1Reco+RECO $name"
    cmsDriver.py RelVal                 --step=$XHLT,RAW2DIGI,L1Reco,RECO                  --conditions=$RTAG --filein=file:RelVal_Raw_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_HLT_RECO_$name.root     --number=$NN $DATAMC --no_exec --datatier 'SIM-RAW-HLT-RECO'               --eventcontent=RAW                     --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_HLT_Reco_$name.py      --processName=$PNAME   $CustomCommand

    echo
    echo "Creating RECO+EI+PAT+VALIDATION+DQM $name"
    cmsDriver.py RelVal                 --step=$STEPS                                      --conditions=$RTAG --filein=file:RelVal_DigiL1RawHLT_$name.root --custom_conditions=$XL1T  --fileout=RelVal_RECO_$name.root         --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-RECO,MINIAODSIM,DQMIO'  --eventcontent=RECOSIM,MINIAODSIM,DQM  --customise=HLTrigger/Configuration/CustomConfigs.Base    $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_RECO_$name.py          --processName=$RNAME  --runUnscheduled


    endif

  end
end
