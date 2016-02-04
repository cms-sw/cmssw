#!/bin/sh

#for FILE in H4-000006100-SM5-COSMIC-STD H4-000006187-SM5-LASER-STD H4-000008203-SM5-PEDESTAL-STD H4-000008205-SM5-TEST_PULSE-MGPA H4-000008206-SM5-COSMIC-STD ; do
#  echo $FILE
#  rm -f /tmp/$USER/$FILE
#  rfcp /castor/cern.ch/cms/archive/ecal/h4/$FILE /tmp/$USER/
#  ln -sf /tmp/$USER/$FILE /tmp/
#done

export CORAL_AUTH_USER=ecal
export CORAL_AUTH_PASSWORD=ecal

#../../../../bin/slc3_ia32_gcc323/testIsolatedElectron

cmsRun miscalib.cfg
#cmsRun tutorial_test.cfg

