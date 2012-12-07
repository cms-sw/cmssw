#! /bin/tcsh

cmsenv

rehash

#
# old files in castor: rfdir /castor/cern.ch/cms/store/...
# new files in eos   : cmsLs /store/...
#

#
# gen sim input files for Monte-Carlo tests
set InputGenSimGRun = /store/relval/CMSSW_5_2_3/RelValProdTTbar/GEN-SIM/START52_V5-v1/0043/D81488D7-0F7A-E111-8BDE-001A92811726.root
set InputGenSimHIon = /store/relval/CMSSW_5_2_7/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM/PU_STARTHI52_V9-v1/0004/52203A21-3708-E211-A85A-003048D2C16E.root
set InputGenSimPIon = /store/relval/CMSSW_5_2_3/RelValProdTTbar/GEN-SIM/START52_V5-v1/0043/D81488D7-0F7A-E111-8BDE-001A92811726.root
#
# lhc raw input files for Real-Data tests
set InputLHCRawGRun = /store/data/Run2012A/MuEG/RAW/v1/000/191/718/14932935-E289-E111-830C-5404A6388697.root
#et InputLHCRawGRun = /store/data/Run2011B/MinimumBias/RAW/v1/000/178/479/3E364D71-F4F5-E011-ABD2-001D09F29146.root
set InputLHCRawHIon = /store/hidata/HIRun2011/HIHighPt/RAW/v1/000/182/838/F20AAF66-F71C-E111-9704-BCAEC532971D.root
set InputLHCRawPIon = /store/data/Run2012A/MuEG/RAW/v1/000/191/718/14932935-E289-E111-830C-5404A6388697.root

#
# global tags to be used for PP and HIon running
set GTAGPPUP = auto:startup
set GTAGPPMC = auto:mc
set GTAGPPRD = auto:hltonline
set GTAGHIUP = auto:starthi
set GTAGHIMC = auto:mc      # MC39_V4HI::All
set GTAGHIRD = auto:hltonline

set NNPPUP = 100
set NNPPMC = 100
set NNPPRD = 100
set NNHIUP = 25
set NNHIMC = 25
set NNHIRD = 25

set XL1T    = "" # syntax: tag,record[,connect,label]
set XL1TPP1 = "" # "L1GtTriggerMenu_L1Menu_Collisions2012_v1_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T"
set XL1TPP2 = "" # "L1GtTriggerMenu_L1Menu_Collisions2012_v2_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T"
set XL1TPP3 = "" # "L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T"
#set XL1TPP3 = "L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc,L1GtTriggerMenuRcd,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2012_v3/sqlFile/L1Menu_Collisions2012_v3_mc.db"
set XL1THI  = "" # "L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2011_v0_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T"
#set XL1THI = "L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2011_v0_mc,L1GtTriggerMenuRcd,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_CollisionsHeavyIons2011_v0/sqlFile/L1Menu_CollisionsHeavyIons2011_v0_mc.db"
#set XL1TPI = "L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2013_v0_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T"
set XL1TPI =  "L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2013_v0_mc,L1GtTriggerMenuRcd,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_CollisionsHeavyIons2013_v0/sqlFile/L1Menu_CollisionsHeavyIons2013_v0_mc.db" 

# specific workflows, first varying the globaltags, then the hlt tables

# Append new JECs (as long as not in GT):
#set XJEC = "JetCorrectorParametersCollection_AK5Calo_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5CaloHLT+JetCorrectorParametersCollection_AK5PF_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5PFHLT+JetCorrectorParametersCollection_AK5PFchs_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5PFchsHLT"
#set XJEC = "JetCorrectorParametersCollection_AK5PF_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5PFHLT+JetCorrectorParametersCollection_AK5PFchs_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5PFchsHLT"

#set XL1TPP1 = ${XL1TPP1}+${XJEC}
#set XL1TPP2 = ${XL1TPP2}+${XJEC}
#set XL1THI  = ${XL1THI}+${XJEC}

foreach gtag ( STARTUP DATA )
#foreach gtag ( DATA STARTUP MC )
  if ( $gtag == DATA ) then
    set GTAG   = $GTAGPPRD
    set GTAGPP = $GTAGPPRD
    set GTAGHI = $GTAGHIRD
    set NNPP   = $NNPPRD
    set NNHI   = $NNHIRD
    set DATAMC = --data
    set PNAME  = HLT1
  else  if ( $gtag == STARTUP ) then
    set GTAG   = $GTAGPPUP
    set GTAGPP = $GTAGPPUP
    set GTAGHI = $GTAGHIUP
    set NNPP   = $NNPPUP
    set NNHI   = $NNHIUP
    set DATAMC = --mc
    set PNAME  = HLT
  else if ( $gtag == MC ) then
    set GTAG   = $GTAGPPMC
    set GTAGPP = $GTAGPPMC
    set GTAGHI = $GTAGHIMC
    set NNPP   = $NNPPMC
    set NNHI   = $NNHIMC
    set DATAMC = --mc
    set PNAME  = HLT
  else
    # unsupported
    continue
  endif

  foreach table ( GRun PIon 5E33v4 7E33v2 7E33v3 7E33v4 8E33v2 HIon )

    set name = ${table}_${gtag}  

    if ( $table == GRun ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:GRun
      set GTAG = ${GTAGPP}_GRun
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun
      set InputLHCRaw = $InputLHCRawGRun
    else if ( $table == 5E33v4 ) then
      set XL1T = $XL1TPP1
      set XHLT = HLT:5E33v4
      set GTAG = ${GTAGPP}_5E33v4
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun
      set InputLHCRaw = $InputLHCRawGRun
    else if ( $table == 7E33v2 ) then
      set XL1T = $XL1TPP1
      set XHLT = HLT:7E33v2
      set GTAG = ${GTAGPP}_7E33v2
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun
      set InputLHCRaw = $InputLHCRawGRun
    else if ( $table == 7E33v3 ) then
      set XL1T = $XL1TPP2
      set XHLT = HLT:7E33v3
      set GTAG = ${GTAGPP}_7E33v3
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun
      set InputLHCRaw = $InputLHCRawGRun
    else if ( $table == 7E33v4 ) then
      set XL1T = $XL1TPP2
      set XHLT = HLT:7E33v4
      set GTAG = ${GTAGPP}_7E33v4
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun
      set InputLHCRaw = $InputLHCRawGRun
    else if ( $table == 8E33v2 ) then
      set XL1T = $XL1TPP3
      set XHLT = HLT:8E33v2
      set GTAG = ${GTAGPP}_8E33v2
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimGRun
      set InputLHCRaw = $InputLHCRawGRun
    else if ( $table == HIon ) then
      set XL1T = $XL1THI
      set XHLT = HLT:HIon
      set GTAG = ${GTAGHI}_HIon
      set NN   = $NNHI
      set SCEN = HeavyIons
      set InputGenSim = $InputGenSimHIon
      set InputLHCRaw = $InputLHCRawHIon
    else if ( $table == PIon ) then
      set XL1T = $XL1TPI
      set XHLT = HLT:PIon
      set GTAG = ${GTAGPP}_PIon
      set NN   = $NNPP
      set SCEN = pp
      set InputGenSim = $InputGenSimPIon
      set InputLHCRaw = $InputLHCRawPIon
    else
      # unsupported
      continue
    endif


    if ( $gtag == DATA ) then

    echo
    echo "Creating L1RePack $name"
    cmsDriver.py RelVal                --step=L1REPACK                             --conditions=$GTAG --filein=$InputLHCRaw                        --custom_conditions=$XL1T --fileout=RelVal_L1RePack_$name.root      --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW'      --eventcontent=RAW          --customise=HLTrigger/Configuration/CustomConfigs.L1T     --scenario=$SCEN --python_filename=RelVal_L1RePack_$name.py

    else

#   echo
#   echo "Creating TTbarGenToHLT $name"
#   cmsDriver.py TTbar_Tauola_8TeV_cfi --step=GEN,SIM,DIGI,L1,DIGI2RAW,$XHLT       --conditions=$GTAG                                              --custom_conditions=$XL1T  --fileout=RelVal_GenSim_$name.root       --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW-HLT'  --eventcontent=FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --scenario=$SCEN --python_filename=RelVal_GenSim_$name.py

    echo
    echo "Creating DigiL1Raw $name"
    cmsDriver.py RelVal                --step=DIGI,L1,DIGI2RAW                     --conditions=$GTAG --filein=$InputGenSim                        --custom_conditions=$XL1T  --fileout=RelVal_DigiL1Raw_$name.root    --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW'      --eventcontent=RAW          --customise=HLTrigger/Configuration/CustomConfigs.L1T     --scenario=$SCEN --python_filename=RelVal_DigiL1Raw_$name.py

    echo
    echo "Creating DigiL1RawHLT $name"
    cmsDriver.py RelVal                --step=DIGI,L1,DIGI2RAW,$XHLT               --conditions=$GTAG --filein=$InputGenSim                        --custom_conditions=$XL1T  --fileout=RelVal_DigiL1RawHLT_$name.root --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RAW-HLT'  --eventcontent=FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --scenario=$SCEN --python_filename=RelVal_DigiL1RawHLT_$name.py  --processName=$PNAME

    if ( ($table != HIon) && ($table != PIon) ) then

    echo
    echo "Creating FastSim $name"
    cmsDriver.py TTbar_Tauola_8TeV_cfi --step GEN,FASTSIM,$XHLT                    --conditions=$GTAG                                              --custom_conditions=$XL1T  --fileout=FastSim_GenToHLT_$name.root    --number=$NN $DATAMC --no_exec --datatier 'GEN-SIM-DIGI-RECO'     --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.FASTSIM --scenario=$SCEN --python_filename=FastSim_GenToHLT_$name.py     --processName=$PNAME

    endif

    endif

    echo
    echo "Creating HLT $name"
    cmsDriver.py RelVal                --step=$XHLT                                --conditions=$GTAG --filein=file:RelVal_Raw_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_HLT_$name.root          --number=$NN $DATAMC --no_exec --datatier 'SIM-DIGI-RAW-HLTDEBUG' --eventcontent=FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --scenario=$SCEN --python_filename=RelVal_HLT_$name.py           --processName=$PNAME

    echo
    echo "Creating HLT2 (re-running HLT) $name"
    cmsDriver.py RelVal                --step=$XHLT                                --conditions=$GTAG --filein=file:RelVal_HLT_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_HLT2_$name.root         --number=$NN $DATAMC --no_exec --datatier 'SIM-DIGI-RAW-HLTDEBUG' --eventcontent=RAW          --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --scenario=$SCEN --python_filename=RelVal_HLT2_$name.py          --processName=HLT2


    if ( $gtag == DATA ) then

    set RTAG = auto:com10_${table}
    if ( $table == PIon ) set RTAG = auto:com10_GRun

    echo
    echo "Creating HLT+RECO $name"
    cmsDriver.py RelVal                --step=$XHLT,RAW2DIGI,L1Reco,RECO           --conditions=$RTAG --filein=file:RelVal_Raw_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_HLT_RECO_$name.root     --number=$NN $DATAMC --no_exec --datatier 'SIM-RAW-HLT-RECO'      --eventcontent=RAW          --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --scenario=$SCEN --python_filename=RelVal_HLT_Reco_$name.py      --processName=$PNAME

    echo
    echo "Creating RECO+DQM $name"
    cmsDriver.py RelVal                --step=RAW2DIGI,L1Reco,RECO,DQM             --conditions=$RTAG --filein=file:RelVal_HLT_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_RECO_$name.root         --number=$NN $DATAMC --no_exec --datatier 'SIM-RAW-HLT-RECO'      --eventcontent=RAW          --customise=HLTrigger/Configuration/CustomConfigs.Base    --scenario=$SCEN --python_filename=RelVal_RECO_$name.py

    else

    set RTAG = $GTAG

    echo
    echo "Creating HLT+RECO $name"
    cmsDriver.py RelVal                --step=$XHLT,RAW2DIGI,L1Reco,RECO           --conditions=$RTAG --filein=file:RelVal_Raw_$name.root          --custom_conditions=$XL1T  --fileout=RelVal_HLT_RECO_$name.root     --number=$NN $DATAMC --no_exec --datatier 'SIM-RAW-HLT-RECO'      --eventcontent=RAW          --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --scenario=$SCEN --python_filename=RelVal_HLT_Reco_$name.py      --processName=$PNAME

    echo
    echo "Creating RECO+VALIDATION+DQM $name"
    cmsDriver.py RelVal                --step=RAW2DIGI,L1Reco,RECO,VALIDATION,DQM  --conditions=$RTAG --filein=file:RelVal_DigiL1RawHLT_$name.root --custom_conditions=$XL1T  --fileout=RelVal_RECO_$name.root         --number=$NN $DATAMC --no_exec --datatier 'SIM-RAW-HLT-RECO'      --eventcontent=RAW          --customise=HLTrigger/Configuration/CustomConfigs.Base    --scenario=$SCEN --python_filename=RelVal_RECO_$name.py

    endif

  end
end
