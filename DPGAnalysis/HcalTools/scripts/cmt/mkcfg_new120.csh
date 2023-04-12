#!/bin/csh
echo "myStart mkcfg_new120.csh"
set j=1
rm -rf PYTHON_${1}
rm -rf TXT_${1}
mkdir PYTHON_${1}
mkdir TXT_${1}

./file_lists.csh ${1}

foreach jj (`cat ${1}`)

set kk=1
cat a.py.beg  > PYTHON_${1}/Reco_${jj}_${kk}_cfg.py
set nn=0

foreach i (`cat TXT_${1}/run_${jj}`)
@ nn = ${nn} + "1"
echo "nn= ${nn} ,  i= ${i},  kk= ${kk}"

#if( ${nn} < "120" ) then
#if( ${nn} < "50" ) then
#if( ${nn} < "30" ) then
#if( ${nn} < "3" ) then

if( ${nn} < "3" ) then

echo "${i}" >> PYTHON_${1}/Reco_${jj}_${kk}_cfg.py

else
echo "${i}" >> PYTHON_${1}/Reco_${jj}_${kk}_cfg.py
cat a.py.end | sed s/Global.root/Global\_${jj}_${kk}.root/g >> PYTHON_${1}/Reco_${jj}_${kk}_cfg.py
@ kk = ${kk} + "1"
cat a.py.beg  > PYTHON_${1}/Reco_${jj}_${kk}_cfg.py
set nn=0
endif

end

if( ${nn} != "0" ) cat a.py.end | sed s/Global.root/Global\_${jj}_${kk}.root/g >> PYTHON_${1}/Reco_${jj}_${kk}_cfg.py
@ j = ${j} + "1"

end

echo "DONE: mkcfg_new120.csh"

######
