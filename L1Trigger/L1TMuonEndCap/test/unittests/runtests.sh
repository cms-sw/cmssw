#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

cd ${LOCAL_TEST_DIR}
rm -f Event*_out.root

# Run 278018
events="1687278667 1541093157 1686648178 1540061587 1686541662 1539957230 1540745931 1687229747"
for event in ${events}; do
  (cmsRun pippo_cfg.py inputFiles=file:Event${event}.root outputFile=Event${event}_out.root >/dev/null 2>&1) || die "Failure processing Event ${event}" $?
  (python test_Event${event}.py) || die "Failure testing Event ${event}" $?
done

# Run 278018 (check ME1/1)
events="1687428853 1687288694 1687130991"
for event in ${events}; do
  (cmsRun pippo_cfg.py inputFiles=file:Event${event}.root outputFile=Event${event}_out.root >/dev/null 2>&1) || die "Failure processing Event ${event}" $?
  (python test_Event${event}.py) || die "Failure testing Event ${event}" $?
done

# Run 281707
#events="135692983 134854523 134963582 300921221 1713517148 135509344 1289966989"
events="135692983 134854523 134963582 300921221 1713517148"
for event in ${events}; do
  (cmsRun pippo_cfg.py inputFiles=file:Event${event}.root outputFile=Event${event}_out.root >/dev/null 2>&1) || die "Failure processing Event ${event}" $?
  (python test_Event${event}.py) || die "Failure testing Event ${event}" $?
done
