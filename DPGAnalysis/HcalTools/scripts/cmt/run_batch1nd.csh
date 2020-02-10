#!/bin/csh

./mkcfg_new.csh ${1}

ls PYTHON_${1}/*py

if( ${status} == "0" ) then
foreach i (`ls PYTHON_${1}`)

set j=`echo ${i} | awk -F _ '{print $2}'`
echo ${i} ${j} ${1}
#./batchjobs.csh ${1} ${j} 

#bsub -q 8nh batchjobs.csh ${1} ${j} `pwd` 
bsub -q 1nd batchjobs.csh ${1} ${j} `pwd`

end
#rm -rf PYTHON_${1}
#rm -rf TXT_${1}
rm Reco_*_cfg.py
else
echo "Problem: No jobs are created: check PYTHON_${1} directory: Notify developpers"
endif

