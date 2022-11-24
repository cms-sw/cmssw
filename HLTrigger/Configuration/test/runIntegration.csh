#!/bin/tcsh

# HLT-integration tests cannot run with SLC6 architectures,
# due to an incompatibility with the latest .jar files of ConfDB-v2.
# For further details, see https://github.com/cms-sw/cmssw/issues/40013#issuecomment-1325571973
if ( "$SCRAM_ARCH" =~ slc6* ) then
  echo
  echo "WARNING -- HLT-integration tests of $2 ($1) will be skipped !"
  echo "           SCRAM_ARCH=$SCRAM_ARCH, and ConfDB access with SLC6 is not supported anymore !"
  echo "           The latest .jar files of ConfDB-v2 are incompatible with scram architectures based on SLC6."
  echo "           For further details, see https://github.com/cms-sw/cmssw/issues/40013#issuecomment-1325571973"
  exit 0
endif

cmsenv
rehash

echo
date +%F\ %a\ %T
echo Start $0 $1 $2

if ( $2 == "" ) then
  set tables = ( GRun )
else if ( $2 == ALL ) then
  set tables = ( GRun HIon PIon PRef 25ns15e33_v4 25ns10e33_v2 Fake Fake1 )
else if ( $2 == IB ) then
  set tables = ( GRun HIon PIon PRef )
else if ( $2 == DEV ) then
  set tables = ( GRun HIon PIon PRef )
else if ( $2 == FULL ) then
  set tables = ( FULL )
else if ( $2 == FAKE ) then
  set tables = ( Fake Fake1 )
else if ( $2 == FROZEN ) then
  set tables = ( 25ns15e33_v4 25ns10e33_v2 Fake Fake1 )
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
