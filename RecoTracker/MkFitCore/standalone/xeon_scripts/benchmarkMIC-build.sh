#! /bin/bash

[ -e "$BIN_DATA_PATH" ] || BIN_DATA_PATH=/data2/slava77/samples/2021/11834.0_TTbar_14TeV+2021/
fin=${BIN_DATA_PATH}/AVE_70_BX01_25ns/memoryFile.fv6.default.211008-c6b7c67.bin

runBenchmark()
{
#    for sV in "sim --cmssw-simseeds" "see --cmssw-stdseeds"; do echo $sV | while read -r sN sO; do
    for sV in "see --cmssw-stdseeds"; do echo $sV | while read -r sN sO; do
            if [ "${1}" == "1" ]; then
                sO="--cmssw-n2seeds"
            fi
            for bV in "BH bh" "STD std" "CE ce"; do echo $bV | while read -r bN bO; do
		    oBase=${base}_${sN}_${bN}
		    for nTH in 1 4 8 16 32; do
		        echo "${oBase}: benchmark [nTH:${nTH}, nVU:8]"
		        time ./mkFit/mkFit --input-file ${fin} --build-${bO} ${sO} --num-thr ${nTH} >& log_${oBase}_NVU8int_NTH${nTH}_benchmark.txt
		    done
                done
            done
        done
    done
}

#cleanup first
make clean
make distclean

make -j 12
export base=SNB_CMSSW_PU70_clean
echo Run default build with base = ${base}
runBenchmark 0

export base=SNB_CMSSW_PU70_clean_cleanSeed
echo Run CLEAN_SEEDS build with base = ${base}
runBenchmark 1
make clean
make distclean


make -j 12 CPPUSERFLAGS+="-march=native -mtune=native" CXXUSERFLAGS+="-march=native -mtune=native"
export base=SNB_CMSSW_PU70_clean_native
echo Run native build with base = ${base}
runBenchmark 0

export base=SNB_CMSSW_PU70_clean_native_cleanSeed
echo Run CLEAN_SEEDS build with base = ${base}
runBenchmark 1
make clean
make distclean

fin10mu=/data2/slava77/samples/2021/10muPt0p2to10HS/memoryFile.fv6.default.211008-c6b7c67.bin

runBenchmark10mu()
{
    for sV in "sim --cmssw-seeds" "see --cmssw-stdseeds"; do echo $sV | while read -r sN sO; do
            if [ "${1}" == "1" ]; then
                sO="--cmssw-n2seeds"
            fi
            for bV in "BH bh" "STD std" "CE ce"; do echo $bV | while read -r bN bO; do
                    oBase=${base}_${sN}_10muPt0p2to10HS_${bN}
                    nTH=8
                    echo "${oBase}: benchmark [nTH:${nTH}, nVU:8]"
                    time ./mkFit/mkFit --input-file ${fin10mu} --build-${bO} ${sO} --num-thr ${nTH} >& log_${oBase}_NVU8int_NTH${nTH}_benchmark.txt
                done
            done
        done
    done

}

#this part has a pretty limited value due to the tiny load in the muon samples
make -j 12
export base=SNB_CMSSW_10mu
echo Run default build with base = ${base}
runBenchmark10mu 1

make clean
make distclean


unset base

