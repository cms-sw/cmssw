#!/bin/bash

dir=${1:-plots}
outdir=${dir}/cmsswval-ttbar-extrectracks
base=SKL-SP_CMSSW_TTbar

echo "Moving plots and text files locally to ${outdir}"
for ttbar in NoPU PU35 PU70 
do
    fulldir=${outdir}/${ttbar}
    mkdir -p ${fulldir}

    mv ${base}_${ttbar}_*.png ${fulldir}
    for build in BH STD CE
    do
	vbase=validation_${base}_${ttbar}_${build}
	mv ${vbase}/totals_${vbase}_cmssw.txt ${fulldir}
    done
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
for ttbar in NoPU PU35 PU70
do
    for build in BH STD CE
    do
	testbase=${base}_${ttbar}_${build}
	rm -rf validation_${testbase}
	rm -rf log_${testbase}_NVU8int_NTH24_cmsswval.txt 
    done
done

rm -rf ${dir}
