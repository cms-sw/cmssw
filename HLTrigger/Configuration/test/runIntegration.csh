#!/bin/tcsh

eval `scram runtime -csh`

echo
date +%F\ %a\ %T
echo Starting $0 $1 $2

if ( $2 == "" ) then
  set tables = ( GRun )
else if ( $2 == ALL ) then
  set tables = ( GRun HIon PIon PRef 2023v11 Fake Fake1 Fake2 )
else if ( $2 == IB ) then
  set tables = ( GRun HIon PIon PRef )
else if ( $2 == DEV ) then
  set tables = ( GRun HIon PIon PRef )
else if ( $2 == FULL ) then
  set tables = ( FULL )
else if ( $2 == FAKE ) then
  set tables = ( Fake Fake1 Fake2 )
else if ( $2 == FROZEN ) then
  set tables = ( 2023v11 )
else
  set tables = ( $2 )
endif

foreach gtag ( $1 )

  if ( $gtag == DATA ) then
    set extraflags = "-x realData=1 -x globalTag=@"
  else
    set extraflags = "-x realData=0 -x globalTag=@"
  endif

  foreach table ( $tables )

    echo
    set name = HLT_Integration_${table}_${gtag}
    touch ${name}
    rm -rf ${name}*

    set infile = file:../RelVal_Raw_${table}_${gtag}.root

    set hltIntegTestCmd = "hltIntegrationTests OnLine_HLT_${table}.py ${extraflags} -d ${name} -i ${infile} -n 100 -j 4 -a cpu"

    echo "`date +%T` ${hltIntegTestCmd} >& ${name}.log"
    time ${hltIntegTestCmd} >& ${name}.log
    set STATUS = $?

    echo "`date +%T` exit status: $STATUS"
    rm -f ${name}/*.root

    if ($STATUS != 0) then
      touch ${name}/issues.txt
      foreach line ("`cat ${name}/issues.txt`")
        cp ${name}/${line}.py  ${name}_${line}.py
        cp ${name}/${line}.log ${name}_${line}.log
      end
    endif

  end

end

echo
echo Finish $0 $1 $2
date +%F\ %a\ %T
