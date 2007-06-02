#!/bin/sh

cd /analysis/sw/CRAB/CMSSW/CMSSW_1_3_0/src
eval `scramv1 runtime -sh`
cd -
cd /analysis/sw/CRAB/RunRate
./Rate.sh
echo -e "... creating histos"

[ ! -e /data1/CrabAnalysis/Rate/History ] && mkdir -p /data1/CrabAnalysis/Rate/History
cp /data1/CrabAnalysis/Rate/RateShort_diff.gif /data1/CrabAnalysis/Rate/History/RateShort_diff_`date +\%Y-\%m-\%d_\%H-\%M-\%S`.gif
cp /data1/CrabAnalysis/Rate/RateShort_integrated.gif /data1/CrabAnalysis/Rate/History/RateShort_integrated_`date +\%Y-\%m-\%d_\%H-\%M-\%S`.gif
cp /data1/CrabAnalysis/Rate/RateShort_logY_diff.gif /data1/CrabAnalysis/Rate/History/RateShort_logY_diff_`date +\%Y-\%m-\%d_\%H-\%M-\%S`.gif
cp /data1/CrabAnalysis/Rate/RateShort_logY_integrated.gif /data1/CrabAnalysis/Rate/History/RateShort_logY_integrated_`date +\%Y-\%m-\%d_\%H-\%M-\%S`.gif

root.exe -b -l -q -x "RunRate.C(\"/data1/CrabAnalysis/Rate/RateShort.txt\",false,true)"
root.exe -b -l -q -x "RunRate.C(\"/data1/CrabAnalysis/Rate/RateShort.txt\",true,true)"
root.exe -b -l -q "RunRate.C(\"/data1/CrabAnalysis/Rate/RateShort.txt\",false,false)"
root.exe -b -l -q "RunRate.C(\"/data1/CrabAnalysis/Rate/RateShort.txt\",true,false)"
cd -
