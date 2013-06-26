#!/bin/bash

eval `scramv1 r -sh`

echo
echo "alca step3:"
echo

grep histograms q5.1.log
grep virtual q5.1.log

