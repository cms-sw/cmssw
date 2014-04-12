#!/bin/bash

eval `scramv1 r -sh`

echo
echo "cosmics reco:"
echo

grep histograms p2.2.log
grep virtual p2.2.log

echo
echo "cosmics harvesting:"
echo

grep histograms q2.2.log
grep virtual q2.2.log

root -l -b -q test2c.C

