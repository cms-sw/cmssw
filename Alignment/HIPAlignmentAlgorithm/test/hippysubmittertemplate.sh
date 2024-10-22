#!/bin/bash

set -euo pipefail

! [ "$(git status --porcelain)" ] || (echo "please commit everything before running"; git status; exit 1)

voms-proxy-info | grep timeleft | grep -v -q 00:00:00 || (echo 'no proxy'; exit 1)

(echo $TMUX > /dev/null) || (echo "run this on a screen"; exit 1)

#hpnumber=
hptype=hp   #or sm
extraopts="--redirectproxy"

#common=Configurations/common_cff_py_pickyours.txt
#lstfile=DataFiles/pickyours.lst
#IOVfile=IOV/RunYYYYYY
#alignmentname=pick a name
#niterations=pick a number

[ -e $lstfile ] || (echo "$lstfile does not exist!"; exit 1)
[ -e $common ] || (echo "$common does not exist!"; exit 1)
[ -e $IOVfile ] || (echo "$IOVfile does not exist!"; exit 1)

commitid=$(git rev-parse HEAD)

git tag $hptype$hpnumber $commitid || (
  status=$?
  if [ $(git rev-parse HEAD) == $(git rev-parse $hptype$hpnumber) ]; then
    exit 0  #from the parentheses
  fi
  echo
  echo "failed to make the tag, see ^"
  echo
  echo "if you previously tried to start the alignment but it failed,"
  echo "you can delete the tag by doing"
  echo
  echo "  git tag --delete $hptype$hpnumber"
  echo
  exit $status
)

submitAndWatchHippy.sh $alignmentname \
                       $niterations \
                       $hptype$hpnumber \
                       $common \
                       Configurations/align_tpl_py.txt \
                       Configurations/TrackSelection \
                       $lstfile \
                       $IOVfile \
                       "$extraopts"
