#!/bin/bash

eval `scramv1 r -sh`

TNUM=150

dqm-mbProfile.py -i 15 -w -f memory_25202.0/performance.json runTheMatrix.py -l 25202.0

dqm-profile.py runTheMatrix.py -l 25.0

if [ $? -ne 0 ]; then
	exit 1
fi
