#!/bin/csh
####
####  ./checkoutput120.csh python_dir output_dir_with_roots 
####      OUTPUT file: missed_run_parts 

ls ${1} > check_${1}

set runold=1
set count=0
rm -rf OUTPUT_${1}
mkdir OUTPUT_${1}

foreach i (`cat check_${1}`)
set run=`echo ${i} | awk -F _ '{print $2}'`  
if( ${run} != ${runold} ) then
set count=0
set runold=${run}
touch OUTPUT_${1}/inputfile_${run}
@ count = ${count} + "1"
echo ${run}"_"${count} >> OUTPUT_${1}/inputfile_${run}
else
@ count = ${count} + "1"
echo ${run}"_"${count} >> OUTPUT_${1}/inputfile_${run}
endif
end
rm missed_run_parts
touch missed_run_parts
foreach i (`ls OUTPUT_${1}`)
foreach j (`cat OUTPUT_${1}/${i}`)
set m=`echo ${j} | awk -F _ '{print $1}'`
set n=`echo ${j} | awk -F _ '{print $2}'`
echo ${i} "${j}" ${m} ${n}
ls ${2}/Global_${m}_${n}.root
if( ${status} != 0 ) then
echo "Reco_"${j}"_cfg.py" >> missed_run_parts
endif
end
end
rm check_${1}

