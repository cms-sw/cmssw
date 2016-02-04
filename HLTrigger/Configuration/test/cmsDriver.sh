#! /bin/tcsh

cmsenv

rehash

#set InputFileGENSIM = file:/scratch/cms/TTbarGenSim3110.root
set InputFileGENSIM = rfio:/castor/cern.ch/user/g/gruen/cms/TTbarGenSim3110.root

# global tags to be used
#set GTAGUP = FrontierConditions_GlobalTag,START36_V1::All
#set GTAGMC = FrontierConditions_GlobalTag,MC_36Y_V1::All
# global stags for PP and HIon running
set GTAGPPUP = auto:startup
set GTAGPPMC = auto:mc
set GTAGHIUP = auto:starthi
set GTAGHIMC = MC39_V4HI::All

set XL1TPP = "" # syntax: tag,record[,connect]
set XL1TPP = "" # "L1GtTriggerMenu_L1Menu_Collisions2011_v1_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T"

# specific workflows, first varying the globaltags, then the hlt tables

foreach gtag ( STARTUP MC )
  if ( $gtag == STARTUP ) then
    set GTAG   = $GTAGPPUP
    set GTAGPP = $GTAGPPUP
    set GTAGHI = $GTAGHIUP
    set XL1THI = ""
  else if ( $gtag == MC ) then
    set GTAG   = $GTAGPPMC
    set GTAGPP = $GTAGPPMC
    set GTAGHI = $GTAGHIMC
    set XL1THI = "L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2010_v2_mc,L1GtTriggerMenuRcd,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_CollisionsHeavyIons2010_v2/sqlFile/L1Menu_CollisionsHeavyIons2010_v2_mc.db"
  else
    # unsupported
    continue
  endif

  # workflows depending ONLY on the globaltag (and not on the HLT table)
  echo
  echo "Creating TTbarGenSim $gtag"
  cmsDriver.py TTbar_Tauola.cfi --step=GEN,SIM                          --conditions=$GTAG                                                                        --fileout=TTbarGenSim_$gtag.root         --number=100 --mc --no_exec --datatier 'GEN-SIM'              --eventcontent=FEVTSIM      --customise=HLTrigger/Configuration/CustomConfigs.Base    --python_filename=TTbarGenSim_$gtag.py

  foreach table ( GRun HIon )
    if ( $table == GRun ) then
      set XL1T = $XL1TPP
      set XHLT = HLT:GRun
      set GTAG = $GTAGPP
    else if ( $table == HIon ) then
      set XL1T = $XL1THI
      set XHLT = HLT:HIon
      set GTAG = $GTAGHI
    else
      # unsupported
      continue
    endif

    set name = ${table}_${gtag}  

    echo
    echo "Creating TTbarGenToHLT $name"
    cmsDriver.py TTbar_Tauola.cfi --step=GEN,SIM,DIGI,L1,DIGI2RAW,$XHLT --conditions=$GTAG                                              --custom_conditions=$XL1T --fileout=TTbarGenHLT_$name.root         --number=100 --mc --no_exec --datatier 'GEN-SIM-DIGI-RAW-HLT' --eventcontent=FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --python_filename=TTbarGenToHLT_$name.py

    echo
    echo "Creating DigiL1Raw $name"
    cmsDriver.py RelVal           --step=DIGI,L1,DIGI2RAW               --conditions=$GTAG --filein=$InputFileGENSIM                    --custom_conditions=$XL1T --fileout=RelVal_DigiL1Raw_$name.root    --number=100 --mc --no_exec --datatier 'GEN-SIM-DIGI-RAW'     --eventcontent=RAW          --customise=HLTrigger/Configuration/CustomConfigs.L1T     --python_filename=RelVal_DigiL1Raw_$name.py

    echo
    echo "Creating DigiL1RawHLT $name"
    cmsDriver.py RelVal           --step=DIGI,L1,DIGI2RAW,$XHLT         --conditions=$GTAG --filein=$InputFileGENSIM                    --custom_conditions=$XL1T --fileout=RelVal_DigiL1RawHLT_$name.root --number=100 --mc --no_exec --datatier 'GEN-SIM-DIGI-RAW-HLT' --eventcontent=FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --python_filename=RelVal_DigiL1RawHLT_$name.py  --processName=HLT

    echo
    echo "Creating HLT $name"
    cmsDriver.py RelVal           --step=$XHLT                          --conditions=$GTAG --filein=file:RelVal_DigiL1Raw_$name.root    --custom_conditions=$XL1T --fileout=RelVal_HLT_$name.root          --number=100 --mc --no_exec --datatier 'RAW-HLT'              --eventcontent=FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --python_filename=RelVal_HLT_$name.py           --processName=HLT

    echo
    echo "Creating HLT2 (re-running HLT) $name"
    cmsDriver.py RelVal           --step=$XHLT                          --conditions=$GTAG --filein=file:RelVal_HLT_$name.root          --custom_conditions=$XL1T --fileout=RelVal_HLT2_$name.root         --number=100 --mc --no_exec --datatier 'RAW-HLT'              --eventcontent=FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --python_filename=RelVal_HLT2_$name.py          --processName=HLT2

#   echo
#   echo "Creating L1HLT2 (re-running L1 and HLT - buggy!) $name"
#   cmsDriver.py RelVal           --step=L1,$XHLT                       --conditions=$GTAG --filein=file:RelVal_DigiL1RawHLT_$name.root --custom_conditions=$XL1T --fileout=RelVal_L1HLT2_$name.root       --number=100 --mc --no_exec --datatier 'GEN-SIM-DIGI-RAW-HLT' --eventcontent=FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT2 --python_filename=RelVal_L1HLT2_$name.py        --processName=HLT2

#   echo
#   echo "Creating RECO $name"
#   cmsDriver.py RelVal           --step=RAW2DIGI,L1Reco,RECO           --conditions=$GTAG --filein=file:RelVal_DigiL1Raw_$name.root    --custom_conditions=$XL1T --fileout=RelVal_RECO_$name.root         --number=100 --mc --no_exec --datatier 'RECO'                 --eventcontent=RECOSIM      --customise=HLTrigger/Configuration/CustomConfigs.Base    --python_filename=RelVal_RECO_$name.py

    echo
    echo "Creating HLT+RECO $name"
    cmsDriver.py RelVal           --step=$XHLT,RAW2DIGI,L1Reco,RECO     --conditions=$GTAG --filein=file:RelVal_DigiL1Raw_$name.root    --custom_conditions=$XL1T --fileout=RelVal_HLT_RECO_$name.root     --number=100 --mc --no_exec --datatier 'RAW-HLT-RECO'         --eventcontent=FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --python_filename=RelVal_HLT_RECO_$name.py      --processName=HLT

#   echo
#   echo "Creating RECO+HLT $name"
#   cmsDriver.py RelVal           --step=RAW2DIGI,L1Reco,RECO,$XHLT     --conditions=$GTAG --filein=file:RelVal_DigiL1Raw_$name.root    --custom_conditions=$XL1T --fileout=RelVal_RECO_HLT_$name.root     --number=100 --mc --no_exec --datatier 'RAW-HLT-RECO'         --eventcontent=FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT  --python_filename=RelVal_RECO_HLT_$name.py      --processName=HLT

  end
end
