#!/bin/bash

eval `scramv1 r -sh`

echo
echo "alca step2:"
echo

grep histograms p6.1.log
grep virtual p6.1.log

