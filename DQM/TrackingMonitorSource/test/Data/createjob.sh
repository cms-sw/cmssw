#!/bin/bash
i=1
for file in `cat ZeroBias1_3.8T_Data_files.txt`
  do
  sed -e "s@INPUTFILES@$file@g" -e "s@INDEX@$i@g" step1_cfg.py.tmpl > "step1_"$i"_cfg.py"
  sed -e "s@INDEX@$i@g" submit.sh.tmpl > "submit_"$i".sh"
  chmod 755 "submit_"$i".sh"
  bsub -q 8nh -J "job"$i < "submit_"$i".sh"
  i=$(($i+1))
done
