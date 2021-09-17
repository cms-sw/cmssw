#!/bin/csh

set j=1
rm -rf PYTHON_${1}
rm -rf TXT_${1}
mkdir PYTHON_${1}
mkdir TXT_${1}

./file_lists.csh ${1}

foreach jj (`cat ${1}`)

cat a.py.beg  > PYTHON_${1}/Reco_${jj}_cfg.py
set nn=0
foreach i (`cat TXT_${1}/run_${jj}`)
@ nn = ${nn} + "1"
echo "nn= ${nn} ,  i= ${i}"
echo "${i}" >> PYTHON_${1}/Reco_${jj}_cfg.py
end

cat a.py.end >> PYTHON_${1}/Reco_${jj}_cfg.py
@ j = ${j} + "1"

end

######
