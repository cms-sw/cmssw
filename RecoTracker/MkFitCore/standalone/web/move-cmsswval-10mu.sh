#!/bin/bash

dir=${1:-plots}
outdir=${dir}/cmsswval-10mu
base=SKL-SP_CMSSW_10mu

echo "Moving plots and text files locally to ${outdir}"
for seed in SimSeed CMSSeed
do
    for region in ECN2 ECN1 BRL ECP1 ECP2 FullDet
    do
	fulldir=${outdir}/${seed}/${region}
	mkdir -p ${fulldir}
 
	srbase=${seed}_${region}
	mv ${base}_${srbase}_*.png ${fulldir}
	for build in BH STD CE
	do
	    vbase=validation_${base}_${srbase}_${build}
	    mv ${vbase}/totals_${vbase}.txt ${fulldir}
	done
    done
    sdir=${outdir}/${seed}
    mv ${sdir}/FullDet/*png ${sdir}/FullDet/*txt ${sdir}
    rm -rf ${sdir}/FullDet
done

host=kmcdermo@lxplus.cern.ch
whost=${host}":~/www"
echo "Moving plots and text files remotely to ${whost}"
scp -r ${dir} ${whost}

echo "Executing remotely ./makereadable.sh ${outdir}"
ssh ${host} bash -c "'
cd www
./makereadable.sh ${outdir}
exit
'"

echo "Removing local files"
for seed in SimSeed CMSSeed
do
    for region in ECN2 ECN1 BRL ECP1 ECP2 FullDet
    do
	srbase=${seed}_${region}
	for build in BH STD CE
	do
	    testbase=${base}_${srbase}_${build}
	    rm -rf validation_${testbase}
	    rm -rf log_${testbase}_NVU8int_NTH24_val.txt 
	done
    done
done

rm -rf ${dir}
