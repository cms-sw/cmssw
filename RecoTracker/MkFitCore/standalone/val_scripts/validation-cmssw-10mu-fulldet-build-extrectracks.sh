#! /bin/bash

make -j 32 WITH_ROOT:=1

dir=/data2/slava77/samples/2021/
subdir=10muPt0p2to10HS/
file=memoryFile.fv6.default.211008-c6b7c67.bin
fin10mu=${dir}/${subdir}/${file}

base=SNB_CMSSW_10mu

for bV in "BH bh" "STD std" "CE ce"
do echo $bV | while read -r bN bO
    do
	oBase=${base}_10muPt0p2to10HS_${bN}
	echo "${oBase}: validation [nTH:32, nVU:32]"
	./mkFit/mkFit --cmssw-n2seeds --cmssw-val-trkparam --input-file ${fin10mu} --build-${bO} --num-thr 32 >& log_${oBase}_NVU32int_NTH32_cmsswval.txt
	mv valtree.root valtree_${oBase}.root
    done
done

make clean

oBase=${base}_10muPt0p2to10HS
for build in BH STD CE
do
    root -b -q -l plotting/runValidation.C\(\"_${oBase}_${build}\",1\)
done
root -b -q -l plotting/makeValidation.C\(\"${oBase}\",\"\",1\)

make distclean
