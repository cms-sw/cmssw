#! /bin/bash

make -j 32 WITH_ROOT:=1

dir=/data2/scratch/toymc
file=simtracks_fulldet_400x2p5k_val.bin

base=SKL-SP_ToyMC_FullDet

for bV in "BH bh" "STD std" "CE ce"
do echo $bV | while read -r bN bO
    do
	oBase=${base}_${bN}
	echo "${oBase}: validation [nTH:32, nVU:32]"
	./mkFit/mkFit --sim-val --read-simtrack-states --seed-input sim --input-file ${dir}/${file} --build-${bO} --num-thr 32 >& log_${oBase}_NVU32int_NTH32_val.txt
	mv valtree.root valtree_${oBase}.root
    done
done

for build in BH STD CE
do
    root -b -q -l plotting/runValidation.C\(\"_SNB_ToyMC_FullDet_${build}\"\)
done
root -b -q -l plotting/makeValidation.C\(\"SNB_ToyMC_FullDet\"\)

make clean
