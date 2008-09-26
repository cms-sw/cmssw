#!/bin/csh
set i = 1
setenv startdir `pwd`
foreach file ( ${startdir}/Run_offline_DQM_*_cfg.py )

mkdir ${startdir}/JOB_${i}
rm submit_${i}.csh
sed "s/NUM/$i/" < submit_template.csh > submit_${i}.csh

bsub -q cmscaf -J job_${i} < submit_${i}.csh

@ i = $i + 1

end
