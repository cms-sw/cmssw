#! /bin/tcsh

cmsenv

rehash

#
# old files in castor: rfdir /castor/cern.ch/cms/store/...
# new files in eos   : cmsLs /store/...
#

#
# gen sim input files for Monte-Carlo tests
set InputGenSimGRun1 = /store/relval/CMSSW_7_1_0/RelValProdTTbar/GEN-SIM/START71_V7-v1/00000/8C172C6B-28FD-E311-8A62-0026189438F4.root
set InputGenSimGRun2 = /store/relval/CMSSW_7_1_9/RelValProdTTbar_13/GEN-SIM/POSTLS171_V17-v1/00000/72B2C6B5-9740-E411-8FF7-002618FDA26D.root
set InputGenSimHIon = /store/relval/CMSSW_5_3_16/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM/PU_STARTHI53_LV1_mar03-v2/00000/143C21CD-E8A2-E311-87BE-0025904C66E8.root
set InputGenSimPIon = /store/relval/CMSSW_7_1_9/RelValProdTTbar_13/GEN-SIM/POSTLS171_V17-v1/00000/72B2C6B5-9740-E411-8FF7-002618FDA26D.root
#
# lhc raw input files for Real-Data tests
set InputLHCRawGRun = /store/data/Run2012A/MuEG/RAW/v1/000/191/718/14932935-E289-E111-830C-5404A6388697.root
#et InputLHCRawGRun = /store/data/Run2011B/MinimumBias/RAW/v1/000/178/479/3E364D71-F4F5-E011-ABD2-001D09F29146.root
set InputLHCRawHIon = /store/hidata/HIRun2011/HIHighPt/RAW/v1/000/182/838/F20AAF66-F71C-E111-9704-BCAEC532971D.root
set InputLHCRawPIon = /store/data/Run2012A/MuEG/RAW/v1/000/191/718/14932935-E289-E111-830C-5404A6388697.root

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

set CustomRun1 = " "
set CustomRun2 = "SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1"
set CustomRun2pp50ns  = "SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1_50ns"
set CustomRun2ppLowPU = "SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1_lowPU"
set CustomRun2HI      = "SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1_HI"
 
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
  else  if ( $gtag == MC ) then
    set BASE1  = $BASE1MC
    set BASE2  = $BASE2MC
    set NNPP   = $NNPPMC
    set NNHI   = $NNHIMC
    set DATAMC = --mc
    set PNAME  = HLT
  else
    # unsupported
    continue
  endif

  foreach table ( GRun 50nsGRun HIon PIon 25nsLowPU LowPU 25ns14e33_v4 25ns14e33_v3 50ns_5e33_v3 25ns14e33_v1 50ns_5e33_v1 Fake )
# foreach table ( GRun 50nsGRun HIon PIon LowPU 25ns14e33_v3 50ns_5e33_v3 Fake )

    set name = ${table}_${gtag}  

    if ( $table == FULL ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:FULL
      set GTAG = ${BASE2}_FULL
      set RTAG = ${BASE2RD}_FULL
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun2
      set InputLHCRaw = $InputLHCRawGRun
      set Custom1 = $CustomRun2
      set Custom2 = " "
      set L1REPACK = L1REPACK:GCTGT
    else if ( $table == Fake ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:Fake
      set GTAG = ${BASE1}_Fake
      set RTAG = ${BASE1RD}_Fake
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun1
      set InputLHCRaw = $InputLHCRawGRun
      set Custom1 = $CustomRun1
      set Custom2 = " "
      set L1REPACK = L1REPACK:GT1
    else if ( $table == GRun ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:GRun
      set GTAG = ${BASE2}_GRun
      set RTAG = ${BASE2RD}_GRun
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun2
      set InputLHCRaw = $InputLHCRawGRun
      set Custom1 = $CustomRun2
      set Custom2 = " "
      set L1REPACK = L1REPACK:GCTGT
    else if ( $table == 25ns14e33_v1 ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:25ns14e33_v1
      set GTAG = ${BASE2}_25ns14e33_v1
      set RTAG = ${BASE2RD}_25ns14e33_v1
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun2
      set InputLHCRaw = $InputLHCRawGRun
      set Custom1 = $CustomRun2
      set Custom2 = " "
      set L1REPACK = L1REPACK:GCTGT
    else if ( $table == 25ns14e33_v3 ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:25ns14e33_v3
      set GTAG = ${BASE2}_25ns14e33_v3
      set RTAG = ${BASE2RD}_25ns14e33_v3
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun2
      set InputLHCRaw = $InputLHCRawGRun
      set Custom1 = $CustomRun2
      set Custom2 = " "
      set L1REPACK = L1REPACK:GCTGT
    else if ( $table == 25ns14e33_v4 ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:25ns14e33_v4
      set GTAG = ${BASE2}_25ns14e33_v4
      set RTAG = ${BASE2RD}_25ns14e33_v4
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun2
      set InputLHCRaw = $InputLHCRawGRun
      set Custom1 = $CustomRun2
      set Custom2 = " "
      set L1REPACK = L1REPACK:GCTGT
    else if ( $table == 50nsGRun ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:50nsGRun
      set GTAG = ${BASE2}_50nsGRun
      set RTAG = ${BASE2RD}_50nsGRun
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun2
      set InputLHCRaw = $InputLHCRawGRun
      set Custom1 = $CustomRun2pp50ns
      set Custom2 = " "
      set L1REPACK = L1REPACK:GCTGT
    else if ( $table == 50ns_5e33_v1 ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:50ns_5e33_v1
      set GTAG = ${BASE2}_50ns_5e33_v1
      set RTAG = ${BASE2RD}_50ns_5e33_v1
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun2
      set InputLHCRaw = $InputLHCRawGRun
      set Custom1 = $CustomRun2pp50ns
      set Custom2 = " "
      set L1REPACK = L1REPACK:GCTGT
    else if ( $table == 50ns_5e33_v3 ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:50ns_5e33_v3
      set GTAG = ${BASE2}_50ns_5e33_v3
      set RTAG = ${BASE2RD}_50ns_5e33_v3
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun2
      set InputLHCRaw = $InputLHCRawGRun
      set Custom1 = $CustomRun2pp50ns
      set Custom2 = " "
      set L1REPACK = L1REPACK:GCTGT
    else if ( $table == HIon ) then
      set XL1T = $XL1THI
      set XHLT = HLT:HIon
      set GTAG = ${BASE2}_HIon
      set RTAG = ${BASE2RD}_HIon
      set NN   = $NNHI
      set SCEN = HeavyIons
      set InputGenSim = $InputGenSimHIon
      set InputLHCRaw = $InputLHCRawHIon
      set Custom1 = $CustomRun2HI
      set Custom2 = " "
      set L1REPACK = L1REPACK:GCTGT
    else if ( $table == PIon ) then
      set XL1T = $XL1TPI
      set XHLT = HLT:PIon
      set GTAG = ${BASE2}_PIon
      set RTAG = ${BASE2RD}_PIon
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimPIon
      set InputLHCRaw = $InputLHCRawPIon
      set Custom1 = $CustomRun2
      set Custom2 = " "
      set L1REPACK = L1REPACK:GCTGT
    else if ( $table == LowPU ) then
      set XL1T = $XL1TLOWPU
      set XHLT = HLT:LowPU
      set GTAG = ${BASE2}_LowPU
      set RTAG = ${BASE2RD}_LowPU
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun2
      set InputLHCRaw = $InputLHCRawGRun
      set Custom1 = $CustomRun2ppLowPU
      set Custom2 = " "
      set L1REPACK = L1REPACK:GCTGT
    else if ( $table == 25nsLowPU ) then
      set XL1T = $XL1TLOWPU
      set XHLT = HLT:25nsLowPU
      set GTAG = ${BASE2}_25nsLowPU
      set RTAG = ${BASE2RD}_25nsLowPU
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun2
      set InputLHCRaw = $InputLHCRawGRun
      set Custom1 = $CustomRun2
      set Custom2 = " "
      set L1REPACK = L1REPACK:GCTGT
    else
      # unsupported
      continue
    endif


    if ( $gtag == DATA ) then

    echo
    echo "Creating L1RePack $name"
    cmsDriver.py RelVal                 --step=$L1REPACK                            --conditions=$GTAG --filein=$InputLHCRaw                        --custom_conditions=$XL1T --fileout=RelVal_L1RePack_$name.root      --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW'      --eventcontent=RAW          --customise=HLTrigger/Configuration/CustomConfigs.L1T     --customise=$Custom1 --customise=$Custom2  --scenario=$SCEN --python_filename=RelVal_L1RePack_$name.py --customise=L1Trigger/Configuration/L1Trigger_custom.customiseResetPrescalesAndMasks

    else

#   echo
#   echo "Creating TTbarGenToHLT $name"
#   cmsDriver.py TTbar_Tauola_13TeV_cfi --step=GEN,SIM,DIGI,L1,DIGI2RAW,$XHLT       --conditions=$GTAG                                              --custom_conditions=$XL1T  --fileout=RelVal_GenSim_$name.root       --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW-HLT'  --eventcontent=FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --customise=$Custom1 --customise=$Custom2  --scenario=$SCEN --python_filename=RelVal_GenSim_$name.py

    echo
    echo "Creating DigiL1Raw $name"
    cmsDriver.py RelVal                 --step=DIGI,L1,DIGI2RAW                     --conditions=$GTAG --filein=$InputGenSim                        --custom_conditions=$XL1T  --fileout=RelVal_DigiL1Raw_$name.root    --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW'      --eventcontent=RAWSIM       --customise=HLTrigger/Configuration/CustomConfigs.L1T     --customise=$Custom1 --customise=$Custom2  --scenario=$SCEN --python_filename=RelVal_DigiL1Raw_$name.py

    echo
    echo "Creating DigiL1RawHLT $name"
    cmsDriver.py RelVal                 --step=DIGI:pdigi_valid,L1,DIGI2RAW,$XHLT   --conditions=$GTAG --filein=$InputGenSim                        --custom_conditions=$XL1T  --fileout=RelVal_DigiL1RawHLT_$name.root --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW-HLT'  --eventcontent=FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --customise=$Custom1 --customise=$Custom2  --scenario=$SCEN --python_filename=RelVal_DigiL1RawHLT_$name.py  --processName=$PNAME

    echo
    echo "Creating FastSim $name"
    cmsDriver.py TTbar_Tauola_13TeV_cfi --step=GEN,SIM,RECOBEFMIX,DIGI,L1,L1Reco,RECO,$XHLT --fast --conditions=$GTAG                               --custom_conditions=$XL1T  --fileout=FastSim_GenToHLT_$name.root    --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RECO'     --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --customise=$Custom1 --customise=$Custom2  --scenario=$SCEN --python_filename=FastSim_GenToHLT_$name.py     --processName=$PNAME

    endif

    echo
    echo "Creating HLT $name"
    cmsDriver.py RelVal                 --step=$XHLT                                --conditions=$GTAG --filein=file:RelVal_Raw_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_HLT_$name.root          --number=$NN $DATAMC --no_exec --datatier 'SIM-DIGI-RAW-HLTDEBUG' --eventcontent=FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --customise=$Custom1 --customise=$Custom2  --scenario=$SCEN --python_filename=RelVal_HLT_$name.py           --processName=$PNAME

    echo
    echo "Creating HLT2 (re-running HLT) $name"
    cmsDriver.py RelVal                 --step=$XHLT                                --conditions=$GTAG --filein=file:RelVal_HLT_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_HLT2_$name.root         --number=$NN $DATAMC --no_exec --datatier 'SIM-DIGI-RAW-HLTDEBUG' --eventcontent=RAW          --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --customise=$Custom1 --customise=$Custom2  --scenario=$SCEN --python_filename=RelVal_HLT2_$name.py          --processName=HLT2


    if ( $gtag == DATA ) then

    echo
    echo "Creating HLT+RECO $name"
    cmsDriver.py RelVal                 --step=$XHLT,RAW2DIGI,L1Reco,RECO           --conditions=$RTAG --filein=file:RelVal_Raw_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_HLT_RECO_$name.root     --number=$NN $DATAMC --no_exec --datatier 'SIM-RAW-HLT-RECO'      --eventcontent=RAW          --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --customise=$Custom1 --customise=$Custom2  --scenario=$SCEN --python_filename=RelVal_HLT_Reco_$name.py      --processName=$PNAME

    echo
    echo "Creating RECO+DQM $name"
    cmsDriver.py RelVal                 --step=RAW2DIGI,L1Reco,RECO,DQM             --conditions=$RTAG --filein=file:RelVal_HLT_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_RECO_$name.root         --number=$NN $DATAMC --no_exec --datatier 'SIM-RAW-HLT-RECO'      --eventcontent=RAW          --customise=HLTrigger/Configuration/CustomConfigs.Base    --customise=$Custom1 --customise=$Custom2  --scenario=$SCEN --python_filename=RelVal_RECO_$name.py

    else

    set RTAG = $GTAG

    echo
    echo "Creating HLT+RECO $name"
    cmsDriver.py RelVal                 --step=$XHLT,RAW2DIGI,L1Reco,RECO           --conditions=$RTAG --filein=file:RelVal_Raw_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_HLT_RECO_$name.root     --number=$NN $DATAMC --no_exec --datatier 'SIM-RAW-HLT-RECO'      --eventcontent=RAW          --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --customise=$Custom1 --customise=$Custom2  --scenario=$SCEN --python_filename=RelVal_HLT_Reco_$name.py      --processName=$PNAME

    echo
    echo "Creating RECO+VALIDATION+DQM $name"
    cmsDriver.py RelVal                 --step=RAW2DIGI,L1Reco,RECO,VALIDATION,DQM  --conditions=$RTAG --filein=file:RelVal_DigiL1RawHLT_$name.root --custom_conditions=$XL1T  --fileout=RelVal_RECO_$name.root         --number=$NN $DATAMC --no_exec --datatier 'SIM-RAW-HLT-RECO'      --eventcontent=RAW          --customise=HLTrigger/Configuration/CustomConfigs.Base    --customise=$Custom1 --customise=$Custom2  --scenario=$SCEN --python_filename=RelVal_RECO_$name.py

    endif

  end
end
