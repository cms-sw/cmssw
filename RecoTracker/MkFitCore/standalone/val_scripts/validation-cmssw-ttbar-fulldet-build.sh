#! /bin/bash

make -j 32 WITH_ROOT:=1

dir=/data2/slava77/samples/2021/11834.0_TTbar_14TeV+2021/
file=memoryFile.fv6.default.211008-c6b7c67.bin

NoPU=AVE_0_BX01_25ns/
PU35=AVE_35_BX01_25ns/
PU50=AVE_50_BX01_25ns/
PU70=AVE_70_BX01_25ns/

base=SKL-SP_CMSSW_TTbar

for ttbar in NoPU PU35 PU50 PU70 
do
    for sV in "SimSeed --cmssw-simseeds" "CMSSeed --cmssw-n2seeds"
    do echo $sV | while read -r sN sO
	do
	    for bV in "BH bh" "STD std" "CE ce"
	    do echo $bV | while read -r bN bO
		do
		    oBase=${base}_${ttbar}_${sN}_${bN}
		    echo "${oBase}: validation [nTH:32, nVU:32]"
		    ./mkFit/mkFit ${sO} --sim-val --input-file ${dir}/${!ttbar}/${file} --build-${bO} --num-thr 32 >& log_${oBase}_NVU32int_NTH32_val.txt
		    mv valtree.root valtree_${oBase}.root
		done
	    done
	done
    done
done

make clean

for ttbar in NoPU PU35 PU50 PU70 
do
    for seed in SimSeed CMSSeed
    do
	oBase=${base}_${ttbar}_${seed}
	for build in BH STD CE
	do
	    root -b -q -l plotting/runValidation.C\(\"_${oBase}_${build}\"\)
	done
	root -b -q -l plotting/makeValidation.C\(\"${oBase}\"\)
    done
done

make distclean
