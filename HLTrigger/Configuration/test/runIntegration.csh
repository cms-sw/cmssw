#!/bin/tcsh

cmsenv
rehash

echo
date +%F\ %a\ %T
echo Start $0 $1 $2

if ( $2 == "" ) then
  set tables = ( GRun 50nsGRun )
else if ( $2 == ALL ) then
  set tables = ( GRun 50nsGRun HIon PIon 25nsLowPU LowPU 25ns14e33_v4 25ns14e33_v3 50ns_5e33_v3 25ns14e33_v1 50ns_5e33_v1 Fake )
else if ( $2 == IB ) then
  set tables = ( GRun 50nsGRun HIon PIon 25nsLowPU LowPU )
else if ( $2 == DEV ) then
  set tables = ( GRun 50nsGRun HIon PIon 25nsLowPU LowPU )
else if ( $2 == LOWPU ) then
  set tables = ( 25nsLowPU LowPU )
else if ( $2 == FULL ) then
  set tables = ( FULL )
else if ( $2 == FAKE ) then
  set tables = ( Fake )
else if ( $2 == FROZEN ) then
  set tables = ( 25ns14e33_v4 25ns14e33_v3 50ns_5e33_v3 25ns14e33_v1 50ns_5e33_v1 Fake )
else
  set tables = ( $2 )
endif

foreach gtag ( $1 )

  if ( $gtag == DATA ) then
    set flags  = ""
    set infix  = hlt
  else
    set flags  = --mc
    set infix  = mc
  endif

  foreach table ( $tables )

    echo
    set name = HLT_Integration_${table}_${gtag}
    touch  ${name}
    rm -rf ${name}*

    set config = `grep tableName OnLine_HLT_${table}.py | cut -f2 -d "'"`
    if ($table == Fake) then
      set basegt = auto:run1_${infix}_${table}
    else 
      set basegt = auto:run2_${infix}_${table}
    endif
    set autogt = "--globaltag=${basegt}"
    set infile = file:../RelVal_Raw_${table}_${gtag}.root

#   -x "--l1-emulator" -x "--l1 L1GtTriggerMenu_L1Menu_Collisions2012_v1_mc" 

    echo "`date +%T` hltIntegrationTests $config -d $name -i $infile -n 100 -j 4 $flags -x ${autogt} -x --type=$table >& $name.log"
    time  hltIntegrationTests $config -d $name -i $infile -n 100 -j 4 $flags -x ${autogt} -x --type=$table >& $name.log
    set STATUS = $?
    echo "`date +%T` exit status: $STATUS"
    rm -f  ${name}/*.root

    if ($STATUS != 0) then
      touch ${name}/issues.txt
      foreach line ("`cat ${name}/issues.txt`")
	cp ${name}/${line}.py   ${name}_${line}.py
	cp ${name}/${line}.log  ${name}_${line}.log
      end
    endif

  end

end

echo
echo Finish $0 $1 $2
date +%F\ %a\ %T
#
