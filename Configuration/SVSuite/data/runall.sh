#!/bin/bash

#
#$Id: runall.sh,v 1.2 2006/12/16 02:15:35 wmtan Exp $
#
#Dummy script to run all integration tests
#
#

testsRecoEgamma="
RecoEgamma_ele_E2000.cfg
RecoEgamma_ele_E50.cfg
RecoEgamma_ele_pt100.cfg
RecoEgamma_ele_pt10.cfg
RecoEgamma_ele_pt35.cfg
RecoEgamma_Hgg_120.cfg
RecoEgamma_HZZ4e_150.cfg
RecoEgamma_pho_E50.cfg
RecoEgamma_pho_pt35.cfg
RecoEgamma_pho_pt50.cfg
RecoEgamma_Zee.cfg
"

testsRecoJets="
RecoJets_Zprime700Dijets.cfg
"

testsMET="
RecoMET_Zjets_Dimuons_300-380.cfg
"

testsTau="
RecoTau_DiTaus_pt_20-420.cfg
"

tests=`echo  $testsRecoEgamma $testsRecoJets $testsMET $testsTau`

report=""

let nfail=0
let npass=0

echo "Tests to be run : " $tests

eval `scramv1 runtime -sh`

for file in $tests 
do
    echo Preparing to run $file
    let starttime=`date "+%s"`
    cmsRun $file
    let exitcode=$?

    let endtime=`date "+%s"`
    let tottime=$endtime-$starttime;   

    if [ $exitcode -ne 0 ] ;then
      echo "cmsRun $file : FAILED - time: $tottime s - exit: $exitcode"
      report="$report \n cmsRun $file : FAILED  - time: $tottime s - exit: $exitcode"
      let nfail+=1
    else 
      echo "cmsRun $file : PASSED - time: $tottime s"
      report="$report \n cmsRun $file : PASSED  - time: $tottime s"
      let npass+=1
    fi 
done


report="$report \n \n $npass tests passed, $nfail failed \n"

echo -e "$report" 
rm -f runall-report.log
echo -e "$report" >& runall-report.log
