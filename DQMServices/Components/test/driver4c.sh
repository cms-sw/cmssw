#!/bin/bash

eval `scramv1 r -sh`

echo
echo "fastsim reco (validation):"
echo

grep histograms p4.1.log
grep virtual p4.1.log

echo
echo "fastsim harvesting (validation):"
echo

grep histograms q4.1.log
grep virtual q4.1.log

echo

