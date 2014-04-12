#!/bin/bash

eval `scramv1 r -sh`

echo
echo "simulation reco (validation_preprod):"
echo

grep histograms p3.1.log
grep virtual p3.1.log

echo
echo "simulation reco (dqm):"
echo

grep histograms p3.2.log
grep virtual p3.2.log

echo
echo "simulation reco (validation_preprod+dqm):"
echo

grep histograms p3.3.log
grep virtual p3.3.log

echo
echo "simulation reco (validation+dqm):"
echo

grep histograms p3.4.log
grep virtual p3.4.log

echo
echo "simulation harvesting (validation_preprod):"
echo

grep histograms q3.1.log
grep virtual q3.1.log

echo
echo "simulation harvesting (dqm):"
echo

grep histograms q3.2.log
grep virtual q3.2.log

echo
echo "simulation harvesting (validation_preprod+dqm):"
echo

grep histograms q3.3.log
grep virtual q3.3.log

echo
echo "simulation harvesting (validation+dqm):"
echo

grep histograms q3.4.log
grep virtual q3.4.log

root -l -b -q test3c.C

