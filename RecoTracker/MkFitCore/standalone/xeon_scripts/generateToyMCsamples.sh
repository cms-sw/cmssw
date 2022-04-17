#! /bin/bash

. data-dir-location.sh

mkdir -p ${dir}

# Building test [1 event in flight]
if [ ! -f ${dir}/simtracks_fulldet_100x2p5k.bin ]; then
    echo "++++Generating 2.5k tracks/event * 100 events for ToyMC building tests with one event in flight++++"
    make -j 12
    ./mkFit/mkFit --num-thr-sim ${n_sim_thr} --num-events 100 --num-tracks 2500 --output-file simtracks_fulldet_100x2p5k.bin
    mv simtracks_fulldet_100x2p5k.bin ${dir}/
    make clean
fi

# Building test [n Events in flight]
if [ ! -f ${dir}/simtracks_fulldet_5kx2p5k.bin ]; then
    echo "++++Generating 2.5k tracks/event * 5k events for ToyMC building tests with nEvents in flight++++"
    make -j 12
    ./mkFit/mkFit --num-thr-sim ${n_sim_thr} --num-events 5000 --num-tracks 2500 --output-file simtracks_fulldet_5kx2p5k.bin
    mv simtracks_fulldet_5kx2p5k.bin ${dir}/
    make clean    
fi

# Validation tests
if [ ! -f ${dir}/simtracks_fulldet_500x2p5k_val.bin ]; then
    echo "++++Generating 2.5k tracks/event * 500 events for ToyMC validation tests++++"
    make -j 12 WITH_ROOT:=1
    ./mkFit/mkFit --num-thr-sim ${n_sim_thr} --sim-val --num-events 500 --num-tracks 2500 --output-file simtracks_fulldet_500x2p5k_val.bin
    mv simtracks_fulldet_500x2p5k_val.bin ${dir}/
    make clean
fi
