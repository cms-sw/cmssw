#!/bin/bash

eval `scramv1 r -sh`

echo
echo "reco (validation):"
echo

grep histograms p1000.1.log
grep virtual p1000.1.log

echo
echo "harvesting (validation):"
echo

grep histograms q1000.1.log
grep virtual q1000.1.log

echo

