#!/bin/sh

eval `scramv1 runtime -sh`

export STAGE_SVCCLASS=cmscaf

hadd Summary_1-50.root *.root > sum_1-50.log &

