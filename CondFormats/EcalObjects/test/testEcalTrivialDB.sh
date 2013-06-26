#!/bin/sh

SETCOLOR_SUCCESS="echo -en \\033[1;32m"
SETCOLOR_FAILURE="echo -en \\033[1;31m"
SETCOLOR_NORMAL="echo -en \\033[0;39m"

echo_success() {
  echo -n " [  "
  $SETCOLOR_SUCCESS
  echo -n $"OK"
  $SETCOLOR_NORMAL
  echo "  ]"
#  echo -ne "\r"
  return 0
}

echo_failure() {
  echo -n "[  "
  $SETCOLOR_FAILURE
  echo -n $"KO"
  $SETCOLOR_NORMAL
  echo "  ]"
#  echo -ne "\r"
  return 0
}

eval `scramv1 runtime -sh`
echo -n "Dumping ECAL fake conditions..."
cmsRun data/print_obj_fake.cfg  > print_obj_fake.stdout 2> print_obj_fake.stderr 
test $? -ne 0 && echo -n "Some problems dumping the conditions. cmsRun return value different from zero" && echo_failure && exit 1 
test `wc -l print_obj_fake.stdout | awk '{print $1}'` -lt 60000 && echo -n "Some problems with the conditions. Dump file too short" && echo_failure && exit 2  
echo_success
echo -n "Dumping ECAL frontier conditions..."
cmsRun data/print_obj_frontier.cfg  > print_obj_frontier.stdout 2> print_obj_frontier.stderr 
test $? -ne 0 && echo -n "Some problems dumping the conditions. cmsRun return value different from zero" && echo_failure && exit 3 
echo_success

DIFF=`diff print_obj_fake.stdout print_obj_frontier.stdout`
if [ "X$DIFF" = "X" ]; then
    echo -n "Fake Conditions correctly loaded into DB" && echo_success
    rm -rf print_obj_fake.stdout print_obj_fake.stderr print_obj_frontier.stdout print_obj_frontier.stderr
    exit 0
else
    echo $DIFF
    echo_failure
    exit 4
fi


