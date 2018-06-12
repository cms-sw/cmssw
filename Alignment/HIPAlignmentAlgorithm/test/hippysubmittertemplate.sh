#!/bin/bash

set -euo pipefail

! [ "$(git status --porcelain)" ] || (echo "please commit everything before running"; git status; exit 1)

voms-proxy-info | grep timeleft | grep -v 00:00:00 || (echo 'no proxy'; exit 1)

(echo $STY) || (echo "run this on a screen"; exit 1)

#hpnumber=
hptype=hp   #or sm
extraopts="--redirectproxy"

#common=Configurations/common_cff_py_pickyours.txt
#lstfile=DataFiles/pickyours.lst
#IOVfile=IOV/RunYYYYYY.dat
#alignmentname=pick a name
#niterations=pick a number

[ -e $lstfile ]
[ -e $common ]
[ -e $IOVfile ]

commitid=$(git rev-parse HEAD)

git tag $hptype$hpnumber $commitid || (
  status=$?
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
