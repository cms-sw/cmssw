#!/bin/bash
set -e
set -x
mkdir -p TTbar_14TeV_TuneCUETP8M1_cfi/root
mkdir -p TTbar_14TeV_TuneCUETP8M1_cfi/raw
mkdir -p TTbar_14TeV_TuneCUETP8M1_cfi/tfr/cand

./generate.sh TTbar_14TeV_TuneCUETP8M1_cfi 1 10
cp pfntuple_1.root TTbar_14TeV_TuneCUETP8M1_cfi/root/

echo "now initialize TF 2.3"
source training_env/bin/activate 
python3 preprocessing.py --input TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_1.root --save-normalized-table --outpath TTbar_14TeV_TuneCUETP8M1_cfi/raw/ --events-per-file 5
python3 tf_data.py --datapath TTbar_14TeV_TuneCUETP8M1_cfi --target gen --num-files-per-tfr 1
python3 tf_model.py --datapath TTbar_14TeV_TuneCUETP8M1_cfi --target gen --ntrain 5 --ntest 5
