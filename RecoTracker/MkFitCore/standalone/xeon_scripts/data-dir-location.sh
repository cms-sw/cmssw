#! /bin/bash

# To be sourced where needed

host=`hostname`

if [[ $host == phi2.t2.* ]]; then
  dir=/data1/scratch/toymc
  n_sim_thr=128
elif [[ $host == phiphi.t2.* ]]; then
  dir=/data/nfsmic/scratch/toymc
  n_sim_thr=12
elif [[ $host == phi3.t2.* ]]; then
  dir=/data2/scratch/toymc
  n_sim_thr=64
else
  dir=/tmp/${USER}/toymc
  n_sim_thr=8
fi
