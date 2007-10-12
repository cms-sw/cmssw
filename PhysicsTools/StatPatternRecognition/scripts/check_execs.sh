#!/bin/bash
#
# Usage: $0 flag(Ascii=1 Root=2)
#
mode=$1
if [ "$mode" != "1" -a "$mode" != "2" ]; then
  echo "Unknown input mode: $mode"
  exit 1
fi
#
input_flag=""
if [ "$mode" == "1" ]; then
  input_flag="-a 1"
fi
input_data="gauss2_uniform_2d_train.pat"
if [ "$mode" == "2" ]; then
  input_data="gauss2_uniform_2d_train_root.pat"
fi
exec_dir="../bin"
echo "TopdownTree 5 0 100" > check_execs.config
#
# check execs
#
echo "$exec_dir  $input_flag    $input_data"
$exec_dir/SprAdaBoostBinarySplitApp $input_flag -n 10 -f 1.spr $input_data
$exec_dir/SprAdaBoostDecisionTreeApp $input_flag -n 10 -l 1000 -f 2.spr $input_data
$exec_dir/SprBaggerApp $input_flag -n 10 -f 3.spr $input_data check_execs.config
$exec_dir/SprBaggerDecisionTreeApp $input_flag -n 10 -l 10 -s 1 -f 4.spr $input_data
$exec_dir/SprBoosterApp $input_flag -n 10 -f 5.spr $input_data check_execs.config
$exec_dir/SprBumpHunterApp $input_flag -n 1000 -v 1 -x 0.5 -f 6.spr $input_data
$exec_dir/SprDecisionTreeApp $input_flag -n 100 -F 7.spr $input_data
$exec_dir/SprFisherLogitApp $input_flag -m 1 -f 8.spr $input_data
$exec_dir/SprFisherLogitApp $input_flag -l -f 9.spr $input_data
$exec_dir/SprStdBackpropApp $input_flag -n 1 -l 10 -N '2:4:2:1' -f 10.spr $input_data
$exec_dir/SprOutputWriterApp $input_flag -A -C '1,2,3,4,5,7,8,9,10' '1.spr,2.spr,3.spr,4.spr,5.spr,7.spr,8.spr,9.spr,10.spr' $input_data save.out
$exec_dir/SprOutputWriterApp $input_flag -C '1,2,3,4,5,7,8,9,10' '1.spr,2.spr,3.spr,4.spr,5.spr,7.spr,8.spr,9.spr,10.spr' $input_data save.root
#
#
#
exit 0
