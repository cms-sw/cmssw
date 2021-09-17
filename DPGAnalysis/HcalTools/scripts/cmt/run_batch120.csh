#!/bin/csh

./mkcfg_new120.csh ${1}

ls PYTHON_${1}/*py

if( ${status} == "0" ) then
foreach i (`ls PYTHON_${1}`)

set j=`echo ${i} | awk -F _ '{print $2}'`
set k=`echo ${i} | awk -F _ '{print $3}'`
echo ${i} ${j} ${k} ${1}
#./batchjobs120.csh ${1} ${j} ${k} `pwd` 

#bsub -q 1nh batchjobs120.csh ${1} ${j} ${k} `pwd` 
bsub -q 8nh batchjobs120.csh ${1} ${j} ${k} `pwd` 
##bsub -q 1nd batchjobs120.csh ${1} ${j} ${k} `pwd`

end
#rm -rf PYTHON_${1}
#rm -rf TXT_${1}
#rm Reco_*_cfg.py
else
echo "Problem: No jobs are created: check PYTHON_${1} directory: Notify developpers"
endif

