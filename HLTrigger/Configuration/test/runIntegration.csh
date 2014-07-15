#!/bin/tcsh

cmsenv
rehash

echo
echo Start $0 $1 $2

if ( $2 == "" ) then
  set tables = ( GRun )
else if ( $2 == ALL ) then
  set tables = ( GRun PIon 2014 HIon )
else if ( $2 == DEV ) then
  set tables = ( GRun PIon HIon )
else if ( $2 == FROZEN ) then
  set tables = ( 2014 )
else
  set tables = ( $2 )
endif

foreach gtag ( $1 )

  foreach table ( $tables )

    if ( $gtag == DATA ) then
      set basepy = OnData
      set basegt = auto:hltonline
      set flags  = ""
    else
      set basepy = OnLine
      if ( $table == HIon ) then
        set basegt = auto:starthi
      else
        set basegt = auto:startup
      endif
      set flags  = --mc
    endif

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
    set STATUS = $?
    echo "exit status: $STATUS"
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
#
