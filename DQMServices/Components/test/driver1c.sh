#!/bin/bash

eval `scramv1 r -sh`

echo
echo "collisions reco:"
echo

grep histograms p1.1.log
grep virtual p1.1.log

echo
echo "collisions harvesting:"
echo

grep histograms q1.1.log
grep virtual q1.1.log

root -l -b -q test1c.C

