#! /bin/tcsh

cmsenv

rehash

#
# old files in castor: rfdir /castor/cern.ch/cms/store/...
# new files in eos   : cmsLs /store/...
#

#
# gen sim input files for Monte-Carlo tests
set InputGenSimGRun1 = /store/relval/CMSSW_7_6_2/RelValProdTTbar/GEN-SIM/76X_mcRun1_realistic_v10-v1/00000/EEEBF25E-229D-E511-8F7D-003048FFCBA8.root
set InputGenSimGRun2 = /store/relval/CMSSW_7_6_2/RelValProdTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v12-v1/00000/88639327-209D-E511-BC0D-0025905A6136.root
set InputGenSimHIon  = /store/relval/CMSSW_7_6_0_pre6/RelValZEEMM_13_HI/GEN-SIM/76X_mcRun2_HeavyIon_v4-v1/00000/EA469164-6A69-E511-A361-008CFA008768.root
set InputGenSimPIon  = $InputGenSimGRun2
#
# lhc raw input files for Real-Data tests
set InputLHCRawGRun1 = /store/data/Run2012A/MuEG/RAW/v1/000/191/718/14932935-E289-E111-830C-5404A6388697.root
set InputLHCRawGRun2 = /store/data/Run2015D/MuonEG/RAW/v1/000/256/677/00000/80950A90-745D-E511-92FD-02163E011C5D.root
set InputLHCRawHIon  = /store/hidata/HIRun2015/HIHardProbes/RAW-RECO/HighPtJet-PromptReco-v1/000/263/689/00000/1802CD9A-DDB8-E511-9CF9-02163E0138CA.root
set InputLHCRawPIon  = $InputLHCRawGRun2

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
set NNHIMC = 25
set NNHIRD = 25

set EraRun1        = " "
set EraRun25ns     = " --era=Run2_25ns "
set EraRun2pp      = " --era=Run2_2016 "
set EraRun2HI      = " --era=Run2_2016,Run2_HI "
 
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
  else  if ( $gtag == MC ) then
    set BASE1  = $BASE1MC
    set BASE2  = $BASE2MC
    set NNPP   = $NNPPMC
    set NNHI   = $NNHIMC
    set DATAMC = --mc
    set PNAME  = HLT
    set RNAME  = RECO
  else
    # unsupported
    continue
  endif

  foreach table ( GRun HIon PIon PRef 25ns10e33_v2 Fake Fake1 )

    set name = ${table}_${gtag}  

    if ( $table == FULL ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:FULL
      set GTAG = ${BASE2}_FULL
      set RTAG = ${BASE2RD}_FULL
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun2
      set InputLHCRaw = $InputLHCRawGRun2
      set Era  = $EraRun2pp
      set Custom = " "
      set L1REPACK = L1REPACK:Full
    else if ( $table == Fake ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:Fake
      set GTAG = ${BASE1}_Fake
      set RTAG = ${BASE1RD}_Fake
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun1
      set InputLHCRaw = $InputLHCRawGRun1
      set Era  = $EraRun1
      set Custom = " "
      set L1REPACK = L1REPACK:GT1
    else if ( $table == Fake1 ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:Fake1
      set GTAG = ${BASE2}_Fake1
      set RTAG = ${BASE2RD}_Fake1
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun2
      set InputLHCRaw = $InputLHCRawGRun2
      set Era  = $EraRun25ns
      set Custom = " "
      set L1REPACK = L1REPACK:GCTGT
    else if ( $table == GRun ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:GRun
      set GTAG = ${BASE2}_GRun
      set RTAG = ${BASE2RD}_GRun
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun2
      set InputLHCRaw = $InputLHCRawGRun2
      set Era  = $EraRun2pp
      set Custom = " "
      set L1REPACK = L1REPACK:Full
    else if ( $table == 25ns10e33_v2 ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:25ns10e33_v2
      set GTAG = ${BASE2}_25ns10e33_v2
      set RTAG = ${BASE2RD}_25ns10e33_v2
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun2
      set InputLHCRaw = $InputLHCRawGRun2
      set Era  = $EraRun2pp
      set Custom = " "
      set L1REPACK = L1REPACK:Full
    else if ( $table == HIon ) then
      set XL1T = $XL1THI
      set XHLT = HLT:HIon
      set GTAG = ${BASE2}_HIon
      set RTAG = ${BASE2RD}_HIon
      set NN   = $NNHI
      set SCEN = HeavyIons
      set InputGenSim = $InputGenSimHIon
      set InputLHCRaw = $InputLHCRawHIon
      set Era  = $EraRun2HI
      set Custom = " "
      set L1REPACK = L1REPACK:Full
    else if ( $table == PIon ) then
      set XL1T = $XL1TPI
      set XHLT = HLT:PIon
      set GTAG = ${BASE2}_PIon
      set RTAG = ${BASE2RD}_PIon
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimPIon
      set InputLHCRaw = $InputLHCRawPIon
      set Era  = $EraRun2pp
      set Custom = " "
      set L1REPACK = L1REPACK:Full
    else if ( $table == PRef ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:PRef
      set GTAG = ${BASE2}_PRef
      set RTAG = ${BASE2RD}_PRef
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun2
      set InputLHCRaw = $InputLHCRawGRun2
      set Era  = $EraRun2pp
      set Custom = " "
      set L1REPACK = L1REPACK:Full
    else
      # unsupported
      continue
    endif


    if ( $gtag == DATA ) then

    echo
    echo "Creating L1RePack $name"
    cmsDriver.py RelVal                 --step=$L1REPACK                                   --conditions=$GTAG --filein=$InputLHCRaw                        --custom_conditions=$XL1T --fileout=RelVal_L1RePack_$name.root      --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW'               --eventcontent=RAW                     --customise=HLTrigger/Configuration/CustomConfigs.L1T     $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_L1RePack_$name.py --customise=L1Trigger/Configuration/L1Trigger_custom.customiseResetPrescalesAndMasks

    else

#   echo
#   echo "Creating TTbarGenToHLT $name"
#   cmsDriver.py TTbar_Tauola_13TeV_cfi --step=GEN,SIM,DIGI,L1,DIGI2RAW,$XHLT              --conditions=$GTAG                                              --custom_conditions=$XL1T  --fileout=RelVal_GenSim_$name.root       --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW-HLT'           --eventcontent=FEVTDEBUGHLT            --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_GenSim_$name.py

    echo
    echo "Creating DigiL1Raw $name"
    cmsDriver.py RelVal                 --step=DIGI,L1,DIGI2RAW                            --conditions=$GTAG --filein=$InputGenSim                        --custom_conditions=$XL1T  --fileout=RelVal_DigiL1Raw_$name.root    --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW'               --eventcontent=RAWSIM                  --customise=HLTrigger/Configuration/CustomConfigs.L1T     $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_DigiL1Raw_$name.py

    echo
    echo "Creating DigiL1RawHLT $name"
    cmsDriver.py RelVal                 --step=DIGI:pdigi_valid,L1,DIGI2RAW,$XHLT          --conditions=$GTAG --filein=$InputGenSim                        --custom_conditions=$XL1T  --fileout=RelVal_DigiL1RawHLT_$name.root --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW-HLT'           --eventcontent=FEVTDEBUGHLT            --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_DigiL1RawHLT_$name.py  --processName=$PNAME

    echo
    echo "Creating FastSim $name"
    cmsDriver.py TTbar_Tauola_13TeV_cfi --step=GEN,SIM,RECOBEFMIX,DIGI,L1,DIGI2RAW,L1Reco,RECO,$XHLT --fast --conditions=$GTAG                             --custom_conditions=$XL1T  --fileout=FastSim_GenToHLT_$name.root    --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RECO'              --eventcontent FEVTDEBUGHLT            --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  $Era --customise=$Custom  --scenario=$SCEN --python_filename=FastSim_GenToHLT_$name.py     --processName=$PNAME

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

    echo
    echo "Creating RECO+EI+PAT+DQM $name"
    cmsDriver.py RelVal                 --step=RAW2DIGI,L1Reco,RECO,EI,PAT,DQM             --conditions=$RTAG --filein=file:RelVal_HLT_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_RECO_$name.root         --number=$NN $DATAMC --no_exec --datatier 'RECO,MINIAOD,DQMIO'             --eventcontent=RECO,MINIAOD,DQM        --customise=HLTrigger/Configuration/CustomConfigs.Base    $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_RECO_$name.py          --processName=$RNAME  --runUnscheduled

    else

    set RTAG = $GTAG

    echo
    echo "Creating HLT+L1Reco+RECO $name"
    cmsDriver.py RelVal                 --step=$XHLT,RAW2DIGI,L1Reco,RECO                  --conditions=$RTAG --filein=file:RelVal_Raw_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_HLT_RECO_$name.root     --number=$NN $DATAMC --no_exec --datatier 'SIM-RAW-HLT-RECO'               --eventcontent=RAW                     --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_HLT_Reco_$name.py      --processName=$PNAME

    echo
    echo "Creating RECO+EI+PAT+VALIDATION+DQM $name"
    cmsDriver.py RelVal                 --step=RAW2DIGI,L1Reco,RECO,EI,PAT,VALIDATION,DQM  --conditions=$RTAG --filein=file:RelVal_DigiL1RawHLT_$name.root --custom_conditions=$XL1T  --fileout=RelVal_RECO_$name.root         --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-RECO,MINIAODSIM,DQMIO'  --eventcontent=RECOSIM,MINIAODSIM,DQM  --customise=HLTrigger/Configuration/CustomConfigs.Base    $Era --customise=$Custom  --scenario=$SCEN --python_filename=RelVal_RECO_$name.py          --processName=$RNAME  --runUnscheduled


    endif

  end
end
