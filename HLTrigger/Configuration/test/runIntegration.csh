#!/bin/tcsh

cmsenv
rehash

echo
echo Start $0 $1 $2

if ( $2 == "" ) then
  set tables = ( GRun )
else if ( $2 == ALL ) then
  set tables = ( GRun PIon 2013 HIon )
else if ( $2 == DEV ) then
  set tables = ( GRun PIon HIon )
else if ( $2 == FROZEN ) then
  set tables = ( 2013 )
else
  set tables = ( $2 )
endif

foreach gtag ( $1 )

  if ( $gtag == DATA ) then
    set basepy = OnData
    set basegt = auto:hltonline
    set flags  = ""
  else
    set basepy = OnLine
    set basegt = auto:startup
    set flags  = --mc
  endif

  foreach table ( $tables )

    echo
    set name = HLT_Integration_${table}_${gtag}
    touch  ${name}
    rm -rf ${name}*

    set config = `grep tableName ${basepy}_HLT_${table}.py | cut -f2 -d "'"`
    set autogt = "--globaltag=${basegt}_${table}"
    set infile = file:../RelVal_Raw_${table}_${gtag}.root

#   -x "--l1-emulator" -x "--l1 L1GtTriggerMenu_L1Menu_Collisions2012_v1_mc" 

    echo "hltIntegrationTests $config -d $name -i $infile -n 100 -j 4 $flags -x ${autogt} >& $name.log"
    time  hltIntegrationTests $config -d $name -i $infile -n 100 -j 4 $flags -x ${autogt} >& $name.log
    echo "exit status: $?"
    rm -f  ${name}/*.root

  end

end

echo
echo Finish $0 $1 $2
#
