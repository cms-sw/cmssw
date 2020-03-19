#!/bin/csh

wget https://cms-service-dqm.web.cern.ch/cms-service-dqm/CAF/certification/Cosmics15/status.Cosmics15.week5.html

set MYDAT=`cat status.Cosmics15.week5.html | grep "<title>" | awk -F \( '{print $2}' | awk -F \) '{print $1}'`
set FIN=`date -d"$MYDAT" '+%Y_%m_%d_%H_%M_%S'`

echo ${FIN}

cat status.Cosmics15.week5.html | grep "<tr><th>" | awk -F "</th><td" '{print $1}' | awk -F "<tr><th>" '{print $2}' | awk -F "</th><th>" '{print $1}' > tmp.htm

touch runsets_${FIN}

set k=0
set j=0
foreach i (`cat tmp.htm`)
if( ${i} == "Run" ) then
set k=0
@ j = ${j} + "1"
else
echo ${i} >> runsets_${FIN}
@ k = ${k} + "1"
endif
if (${j} > "1") then
break
endif 
end

rm 
tmp.htm
