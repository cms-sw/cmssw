#!/bin/sh

export CORAL_AUTH_USER=ecal
export CORAL_AUTH_PASSWORD=ecal

rm -f ecalMiscalib.db

../../../../bin/slc3_ia32_gcc323/WriteEcalMiscalibConstants

#cmsRun monitor.txt
#cmsRun tutorial_test.cfg

