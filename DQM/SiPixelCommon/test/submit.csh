#!/bin/csh
set i = 1
setenv startdir `pwd`
foreach file ( ${startdir}/Run_offline_DQM_*_cfg.py )

if(! -d ${startdir}/JOB_${i} ) then
mkdir ${startdir}/JOB_${i}
endif

if( -e submit_${i}.csh ) then
rm submit_${i}.csh
endif

sed "s/NUM/${i}/" < submit_template.csh > submit_${i}.csh
sed "s#CFGDIR#${startdir}#" < submit_${i}.csh > tmp.csh
mv tmp.csh submit_${i}.csh


 bsub -q cmscaf -J job_${i} < submit_${i}.csh

@ i = $i + 1

end
