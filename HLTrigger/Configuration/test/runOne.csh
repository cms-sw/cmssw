#!/bin/tcsh

cmsenv
rehash

set rawLHC = L1RePack
set rawSIM = DigiL1Raw

echo
echo Starting $0 $1 $2

if ( $2 == "" ) then
  set tables = ( GRun )
else if ( ($2 == all) || ($2 == ALL) ) then
  set tables = ( GRun PIon 2013 HIon )
else if ( ($2 == dev) || ($2 == DEV) ) then
  set tables = ( GRun PIon HIon )
else if ( ($2 == frozen) || ($2 == FROZEN) ) then
  set tables = ( 2013 )
else
  set tables = ( $2 )
endif

foreach gtag ( $1 )

  foreach table ( $tables )

    if ($gtag == DATA) then
      set base = RelVal_${rawLHC}
    else
      set base = RelVal_${rawSIM}
    endif

#   run workflows

    set base = ( $base ONLINE_HLT RelVal_HLT RelVal_HLT2 )

    if ( $gtag == STARTUP ) then
      if ( ( $table != HIon ) && ( $table != PIon) ) then
        set base = ( $base FastSim_GenToHLT )
      endif
    endif

    foreach task ( $base )

      echo
      set name = ${task}_${table}_${gtag}
      rm -f $name.{log,root}
      echo "cmsRun $name.py >& $name.log"
#     ls -l        $name.py
      time  cmsRun $name.py >& $name.log
      echo "exit status: $?"

      if ( ( $task == RelVal_${rawLHC} ) || ( $task == RelVal_${rawSIM} ) ) then
#       link to input file for subsequent steps
        rm -f              RelVal_Raw_${table}_${gtag}.root
        ln -s ${name}.root RelVal_Raw_${table}_${gtag}.root
      endif

    end

  end

end

# special fastsim integration test

if ( $1 == STARTUP ) then
  foreach task ( IntegrationTestWithHLT_cfg )

    echo
    set name = ${task}
    rm -f $name.{log,root}
    echo "cmsRun $name.py >& $name.log"
#   ls -l        $name.py
    time  cmsRun $name.py >& $name.log
    echo "exit status: $?"

  end
endif

# separate hlt+reco and reco+(validation)+dqm workflows

foreach gtag ( $1 )

  foreach table ( $tables )

    if ($gtag == DATA) then
      set base = ( RelVal_HLT_Reco                     RelVal_RECO )
    else
      set base = ( RelVal_HLT_Reco RelVal_DigiL1RawHLT RelVal_RECO )
    endif

    foreach task ( $base )

      echo
      set name = ${task}_${table}_${gtag}
      rm -f $name.{log,root}
      echo "cmsRun $name.py >& $name.log"
#     ls -l        $name.py
      time  cmsRun $name.py >& $name.log
      echo "exit status: $?"

    end

  end

end

# running each HLT trigger path individually one by one

if ( ($2 != all) && ($2 != dev) && ($2 != frozen) ) then
  ./runIntegration.csh $1 $2
endif

echo
echo Finished $0 $1 $2
#
