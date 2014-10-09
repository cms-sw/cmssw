#!/bin/tcsh

cmsenv
rehash

echo
date +%F\ %a\ %T
echo
echo "Existing cfg files:"
ls -l On{Data,Mc}*.py

echo
echo "Creating OnLine cfg files adding the HLTAnalyzerEndpath:"

foreach gtag ( Data Mc )
  set GTAG = ` echo $gtag | tr "[a-z]" "[A-Z]" `
  foreach table ( FULL Fake 2014 GRun HIon PIon )
    set oldfile = On${gtag}_HLT_${table}.py
    set newfile = OnLine_HLT_${table}_${GTAG}.py
    rm -f $newfile
    cp $oldfile $newfile
    cat >> $newfile <<EOF
#
if not ('HLTAnalyzerEndpath' in process.__dict__) :
    from HLTrigger.Configuration.HLT_FULL_cff import hltL1GtTrigReport,hltTrigReport
    process.hltL1GtTrigReport = hltL1GtTrigReport
    process.hltTrigReport = hltTrigReport
    process.hltTrigReport.HLTriggerResults = cms.InputTag( 'TriggerResults','',process.name_() )
    process.HLTAnalyzerEndpath = cms.EndPath(process.hltL1GtTrigReport + process.hltTrigReport)
#
EOF
  end
end

echo
echo "Created OnLine cfg files:"
ls -l OnLine*.py

echo
echo "Creating offline cfg files with cmsDriver"
echo "./cmsDriver.csh"
time  ./cmsDriver.csh

echo
echo "Creating special FastSim IntegrationTestWithHLT"

foreach task ( IntegrationTestWithHLT_cfg )
  echo
  set name = ${task}
  rm -f $name.py

  if ( -f $CMSSW_BASE/src/FastSimulation/Configuration/test/$name.py ) then
    cp         $CMSSW_BASE/src/FastSimulation/Configuration/test/$name.py $name.py
  else
    cp $CMSSW_RELEASE_BASE/src/FastSimulation/Configuration/test/$name.py $name.py
  endif
  ls -l $name.py
end

echo
date +%F\ %a\ %T
echo
echo "Running selected cfg files from:"
pwd

rm -f                           ./runOne.log 
time ./runOne.csh DATA    $1 >& ./runOne.log &
time ./runOne.csh MC      $1

  set N = 0
  cp -f ./runOne.log ./runOne.tmp  
  grep -q Finished   ./runOne.tmp
  set F = $?

while ( $F )
  awk "{if (NR>$N) {print}}"  ./runOne.tmp
  set N = `cat ./runOne.tmp | wc -l`
  sleep 13
  cp -f ./runOne.log ./runOne.tmp  
  grep -q Finished   ./runOne.tmp
  set F = $?
end

wait

  awk "{if (NR>$N) {print}}"  ./runOne.log
  rm -f ./runOne.{log,tmp}

echo
echo "Resulting log files:"
ls -l *.log
echo
date +%F\ %a\ %T
