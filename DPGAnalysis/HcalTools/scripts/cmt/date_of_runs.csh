#!/bin/csh
if (${1} == "") then
echo "No input run list. Please use command: ./run_date.csh runsets"
exit
endif
touch runsets_1
foreach i (`cat $1`)
echo ${i}
./das_client.py --query="run=${i} | grep run.start_time" | grep 20 > tmp
set DATE=`cat tmp`
echo ${i} ${DATE} >> runsets_1
rm tmp
end
