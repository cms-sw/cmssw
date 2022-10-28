#!/bin/bash

dir=${1:-plots}
outdir=${dir}/cmsswval-10mu-extrectracks
base=SKL-SP_CMSSW_10mu

echo "Moving plots and text files locally to ${outdir}"
for region in ECN2 ECN1 BRL ECP1 ECP2 FullDet
do
    fulldir=${outdir}/${region}
    mkdir -p ${fulldir}
    
    mv ${base}_${region}_*.png ${fulldir}
    for build in BH STD CE
    do
	vbase=validation_${base}_${region}_${build}
	mv ${vbase}/totals_${vbase}_cmssw.txt ${fulldir}
    done
done

mv ${outdir}/FullDet/*png ${outdir}/FullDet/*txt ${outdir}
rm -rf ${outdir}/FullDet

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
for region in ECN2 ECN1 BRL ECP1 ECP2 FullDet
do
    for build in BH STD CE
    do
	testbase=${base}_${region}_${build}
	rm -rf validation_${testbase}
	rm -rf log_${testbase}_NVU8int_NTH24_cmsswval.txt 
    done
done

rm -rf ${dir}
